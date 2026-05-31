// Distributed per-layer UNet split-API correctness test (SD1.5).
//
// Proves the public "Distributed per-layer UNet execution" surface in
// include/stable-diffusion.h is correct by comparing, in-process:
//   A) a monolithic full-range eval   sd_compute_unet_split_range(0, count)
//   B) a chained 2-stage eval         [0, CUT) -> carry -> [CUT, count)
// for a single denoise step with deterministic inputs. The split path is
// documented to be the same code walked in pieces, so A and B must agree.
//
// Correctness discriminator (project memory cf12-w7-nway-split):
//   "trivial cuts are BIT-EXACT". A cut at [0,1)/[1,25) (and [0,24)/[24,25))
//   carries only hs[0]=conv_in (fp32) plus h/emb and must round-trip the carry
//   plumbing EXACTLY. If the trivial cut is ~0 the get/set/thread carry path is
//   correct and any residual at an interior cut is genuine fp16-boundary drift,
//   not a test/carry bug. If the trivial cut is NOT ~0, there is a carry-
//   threading bug to fix. We use the trivial cut as the real pass criterion and
//   report a SWEEP {1, 13, 24} so the boundary-error profile is visible.
//
// This mirrors the proven reference validator
//   llama-distributed/python/dpp_runtime/validate_nway_split.py
// which builds x as a seeded ~N(0,1) latent of WHCN shape (64,64,4,1)
// (_encode_step_x_frame) at a fixed (step_idx, timestep) and chains the
// {h, hs, emb} carry between contiguous block ranges (_run_chain). To stay in a
// representative regime we drive BOTH x and context from seeded randn (the
// validator's context is real text-encoder output; a fixed-fill context pushes
// cross-attention into a degenerate regime and inflates boundary error).
//
// Build (optional target, off by default):
//   cmake -DSD_BUILD_SPLIT_TEST=ON ...   ->   target sd-test-unet-split
// Run:
//   sd-test-unet-split [model.gguf]      (or env DIST_SDCPP_MODEL)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "stable-diffusion.h"

namespace {

// ── SD1.5 input contract (WHCN; see validator + unet.hpp) ──────────────────
//   x         : latent, WHCN (64, 64, 4, 1)   — validate_nway_split.py
//                                                _encode_step_x_frame((64,64,4,1))
//   timesteps : (1)                            — single UNet timestep t
//   context   : CLIP crossattn, WHCN (768, 77, 1, 1)
//                768 = context_dim, 77 = n_context tokens (unet.hpp:180)
//   y         : absent for SD1.5 (adm/label_emb is SDXL-only, unet.hpp:360)
constexpr int64_t kLatentW = 64;
constexpr int64_t kLatentH = 64;
constexpr int64_t kLatentC = 4;
constexpr int64_t kLatentN = 1;

constexpr int64_t kCtxDim    = 768;  // context_dim (SD1.5)
constexpr int64_t kCtxTokens = 77;   // n_context

constexpr int kExpectedBlocks = 25;  // SD1.5/SD2 linear block count

// Gates. The TRIVIAL cut exercises only the carry plumbing (hs[0]=conv_in,
// h, emb — all fp32) and must round-trip near-exactly; this is the real pass
// criterion. The MID cut additionally crosses fp16 block boundaries, so its
// residual is a (documented) drift budget, not a correctness signal.
constexpr float kTrivialTol = 1e-3f;   // trivial cut must be ~bit-exact
constexpr float kMidTol     = 0.12f;   // fp16-boundary budget (validator ~6%)

// Deterministic seeded ~N(0,1) fill, matching the validator's
// random.Random(seed).gauss(0,1) intent (values, not exact bitstream).
std::vector<float> randn_fill(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

bool stage_inputs(sd_split_state_t* st) {
    // x: seeded latent (realistic unit-scale signal so rel-error is meaningful).
    const int64_t x_shape[4] = {kLatentW, kLatentH, kLatentC, kLatentN};
    const size_t  x_numel    = (size_t)kLatentW * kLatentH * kLatentC * kLatentN;
    std::vector<float> x = randn_fill(x_numel, /*seed=*/7);
    if (sd_split_state_set_input(st, "x", x.data(), x_shape, 4) != SD_SPLIT_OK) {
        fprintf(stderr, "set_input(x) failed\n");
        return false;
    }

    // timesteps: a single fixed t.
    const int64_t t_shape[1] = {1};
    const float   t_val[1]   = {500.0f};
    if (sd_split_state_set_input(st, "timesteps", t_val, t_shape, 1) != SD_SPLIT_OK) {
        fprintf(stderr, "set_input(timesteps) failed\n");
        return false;
    }

    // context: seeded randn crossattn embedding. randn (vs a fixed fill) keeps
    // cross-attention in a representative, non-degenerate regime closer to the
    // real text-encoder output the validator uses, so the boundary error we
    // measure reflects the deployed pipeline rather than a synthetic artifact.
    const int64_t c_shape[4] = {kCtxDim, kCtxTokens, 1, 1};
    const size_t  c_numel    = (size_t)kCtxDim * kCtxTokens;
    std::vector<float> ctx = randn_fill(c_numel, /*seed=*/13);
    if (sd_split_state_set_input(st, "context", ctx.data(), c_shape, 4) != SD_SPLIT_OK) {
        fprintf(stderr, "set_input(context) failed\n");
        return false;
    }

    // y: SD1.5 has no label/vector input — pass the documented null triple.
    if (sd_split_state_set_input(st, "y", nullptr, nullptr, 0) != SD_SPLIT_OK) {
        fprintf(stderr, "set_input(y) failed\n");
        return false;
    }
    return true;
}

// Copy the noise-pred out of a state into an owned buffer (the API returns a
// borrowed pointer valid only until the next mutating call).
bool copy_output(const sd_split_state_t* st, std::vector<float>& out) {
    const float*   data  = nullptr;
    const int64_t* shape = nullptr;
    int            ndims = 0;
    if (sd_split_state_get_output(st, &data, &shape, &ndims) != SD_SPLIT_OK) {
        fprintf(stderr, "get_output failed\n");
        return false;
    }
    size_t numel = 1;
    for (int i = 0; i < ndims; ++i) numel *= (size_t)shape[i];
    out.assign(data, data + numel);
    return true;
}

// Thread the {h, hs[], emb} carry from a producer state (after [lo,CUT)) onto a
// consumer state (before [CUT,hi)) using ONLY the documented carry API. In a
// real deployment producer/consumer live on different rigs; here we use two
// distinct state objects so the test exercises the get/set carry path rather
// than reusing one object's internal carry.
//
// `verbose` prints the hs_count once and, on the first carry probe, the WHCN
// shape of hs[0] (conv_in). The public surface delivers carry as "plain
// contiguous fp32" (stable-diffusion.h "Tensor layout"), so the on-the-wire
// ggml_type is always fp32 and is not reachable here; the fp16 boundary, if
// any, is internal to sd_compute_unet_split_range and noted in the writeup.
bool thread_carry(const sd_split_state_t* src, sd_split_state_t* dst, bool verbose) {
    int hs_count = 0;
    if (sd_split_state_get_carry_count(src, &hs_count) != SD_SPLIT_OK) {
        fprintf(stderr, "get_carry_count failed\n");
        return false;
    }
    if (verbose) printf("[split] carry hs_count=%d\n", hs_count);
    if (sd_split_state_set_hs_count(dst, hs_count) != SD_SPLIT_OK) {
        fprintf(stderr, "set_hs_count failed\n");
        return false;
    }

    bool printed_shape = false;
    auto move_one = [&](const char* name) -> bool {
        const float*   data  = nullptr;
        const int64_t* shape = nullptr;
        int            ndims = 0;
        if (sd_split_state_get_carry_tensor(src, name, &data, &shape, &ndims) != SD_SPLIT_OK) {
            fprintf(stderr, "get_carry_tensor(%s) failed\n", name);
            return false;
        }
        if (verbose && !printed_shape && std::strncmp(name, "hs.0", 4) == 0) {
            printf("[split] carry %s ndims=%d shape=[", name, ndims);
            for (int i = 0; i < ndims; ++i) printf("%s%lld", i ? "," : "",
                                                    (long long)shape[i]);
            printf("] dtype=fp32 (public carry surface; internal boundary may be fp16)\n");
            printed_shape = true;
        }
        if (sd_split_state_set_carry_tensor(dst, name, data, shape, ndims) != SD_SPLIT_OK) {
            fprintf(stderr, "set_carry_tensor(%s) failed\n", name);
            return false;
        }
        return true;
    };

    if (!move_one("h")) return false;
    if (!move_one("emb")) return false;
    for (int i = 0; i < hs_count; ++i) {
        char nm[32];
        std::snprintf(nm, sizeof(nm), "hs.%d", i);
        if (!move_one(nm)) return false;
    }
    return true;
}

struct CutResult {
    int    cut      = 0;
    int    hs_count = 0;
    double max_abs  = 0.0;
    double rel_rms  = 0.0;
    bool   ok       = false;   // executed and shapes matched
};

// Run the chained 2-stage eval [0,cut) -> carry -> [cut,count) and compare its
// noise-pred to the full-range reference. Returns metrics in `res`.
CutResult run_2stage(sd_ctx_t* ctx, int cut, int count,
                     const std::vector<float>& ref, bool verbose) {
    CutResult res;
    res.cut = cut;

    sd_split_state_t* st_a = sd_split_state_new();
    sd_split_state_t* st_b = sd_split_state_new();
    std::vector<float> out_split;

    do {
        if (st_a == nullptr || !stage_inputs(st_a)) break;
        int r = sd_compute_unet_split_range(ctx, 0, cut, /*step_idx=*/0,
                                            /*total_steps=*/1, st_a);
        if (r != SD_SPLIT_OK) {
            fprintf(stderr, "stage0 [0,%d) rc=%d\n", cut, r);
            break;
        }
        if (st_b == nullptr || !stage_inputs(st_b)) break;

        int hs_count = 0;
        sd_split_state_get_carry_count(st_a, &hs_count);
        res.hs_count = hs_count;

        if (!thread_carry(st_a, st_b, verbose)) break;
        r = sd_compute_unet_split_range(ctx, cut, count, /*step_idx=*/0,
                                        /*total_steps=*/1, st_b);
        if (r != SD_SPLIT_OK) {
            fprintf(stderr, "stage1 [%d,%d) rc=%d\n", cut, count, r);
            break;
        }
        if (!copy_output(st_b, out_split)) break;

        if (out_split.size() != ref.size() || ref.empty()) {
            fprintf(stderr, "numel mismatch ref=%zu split=%zu (cut=%d)\n",
                    ref.size(), out_split.size(), cut);
            break;
        }
        double sum_sq_diff = 0.0, sum_sq_ref = 0.0, max_abs = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double a = ref[i], b = out_split[i];
            const double d = std::fabs(a - b);
            if (d > max_abs) max_abs = d;
            sum_sq_diff += d * d;
            sum_sq_ref  += a * a;
        }
        res.max_abs = max_abs;
        res.rel_rms = std::sqrt(sum_sq_diff /
                                (sum_sq_ref > 1e-12 ? sum_sq_ref : 1e-12));
        res.ok = true;
    } while (false);

    if (st_b) sd_split_state_free(st_b);
    if (st_a) sd_split_state_free(st_a);
    return res;
}

}  // namespace

int main(int argc, char** argv) {
    const char* model_path = (argc > 1) ? argv[1] : std::getenv("DIST_SDCPP_MODEL");
    if (model_path == nullptr || model_path[0] == '\0') {
        fprintf(stderr,
                "usage: %s <model.gguf>   (or set DIST_SDCPP_MODEL)\n", argv[0]);
        return 2;
    }
    printf("[split] model=%s\n", model_path);

    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path             = model_path;
    params.vae_decode_only        = false;
    params.free_params_immediately = false;
    // n_threads is left at the sd_ctx_params_init default (physical cores).

    sd_ctx_t* ctx = new_sd_ctx(&params);
    if (ctx == nullptr) {
        fprintf(stderr, "new_sd_ctx failed for %s\n", model_path);
        return 1;
    }

    int rc = 1;
    sd_split_state_t* st_full = nullptr;

    do {
        const char* tag = sd_loaded_backbone_tag(ctx);
        const int   count = sd_unet_block_count(ctx);
        printf("[split] backbone=%s block_count=%d (expected %d)\n",
               tag ? tag : "(null)", count, kExpectedBlocks);
        if (count != kExpectedBlocks) {
            fprintf(stderr,
                    "FAIL: sd_unet_block_count=%d, expected %d for SD1.5\n",
                    count, kExpectedBlocks);
            printf("RESULT: FAIL\n");
            break;
        }

        // ── A: monolithic full-range [0, count) reference ───────────────────
        st_full = sd_split_state_new();
        if (st_full == nullptr || !stage_inputs(st_full)) break;
        int r = sd_compute_unet_split_range(ctx, 0, count, /*step_idx=*/0,
                                            /*total_steps=*/1, st_full);
        if (r != SD_SPLIT_OK) {
            fprintf(stderr, "full-range sd_compute_unet_split_range rc=%d\n", r);
            break;
        }
        std::vector<float> out_full;
        if (!copy_output(st_full, out_full)) break;
        printf("[split] A full-range  numel=%zu\n", out_full.size());

        // ── Trivial-cut PROBE: [0,1) -> [1,count) ───────────────────────────
        // Carries only hs[0]=conv_in (+ h, emb), all fp32 — no fp16 block
        // boundary crossed. This isolates the carry get/set/thread plumbing.
        // If trivial.rel_rms ~ 0, the plumbing is correct and any interior-cut
        // residual is genuine fp16-boundary drift, NOT a carry bug.
        printf("[split] --- trivial-cut probe [0,1)/[1,%d) ---\n", count);
        CutResult trivial = run_2stage(ctx, /*cut=*/1, count, out_full, /*verbose=*/true);
        if (!trivial.ok) break;
        printf("[split] TRIVIAL cut=1 hs_count=%d max_abs_diff=%.6e rel_rms=%.6e (%.4f%%)\n",
               trivial.hs_count, trivial.max_abs, trivial.rel_rms,
               trivial.rel_rms * 100.0);

        // ── Boundary-error SWEEP: cuts {1, 13, 24} ──────────────────────────
        // cut=1  : trivial (conv_in carry only)
        // cut=13 : legacy middle-block boundary (deepest channels, most hs)
        // cut=24 : trivial at the top (carry everything but the final out-conv)
        printf("[split] --- sweep cut,hs_count,max_abs_diff,rel_rms ---\n");
        const int sweep_cuts[3] = {1, 13, 24};
        CutResult sweep[3];
        bool sweep_ok = true;
        for (int i = 0; i < 3; ++i) {
            sweep[i] = (sweep_cuts[i] == 1)
                           ? trivial
                           : run_2stage(ctx, sweep_cuts[i], count, out_full, /*verbose=*/false);
            if (!sweep[i].ok) { sweep_ok = false; break; }
            printf("[split] SWEEP cut=%-2d hs_count=%-2d max_abs_diff=%.6e rel_rms=%.6e (%.4f%%)\n",
                   sweep[i].cut, sweep[i].hs_count, sweep[i].max_abs,
                   sweep[i].rel_rms, sweep[i].rel_rms * 100.0);
        }
        if (!sweep_ok) break;

        // mid cut (13) is index 1 of the sweep.
        const CutResult& mid = sweep[1];

        // ── Gate ────────────────────────────────────────────────────────────
        // Real pass criterion: the TRIVIAL cut is near-exact (carry plumbing
        // correct). Secondary: the MID cut stays within the fp16-boundary
        // budget. A trivial-exact + over-budget mid would point to fp16 drift /
        // input regime, not a carry bug — reported explicitly either way.
        const bool trivial_pass = trivial.rel_rms < kTrivialTol;
        const bool mid_pass     = mid.rel_rms     < kMidTol;
        const bool pass         = trivial_pass && mid_pass;

        printf("[split] GATE trivial=%s (rel_rms=%.6e < %.0e) "
               "mid=%s (rel_rms=%.4f < %.2f)\n",
               trivial_pass ? "PASS" : "FAIL", trivial.rel_rms, (double)kTrivialTol,
               mid_pass ? "PASS" : "FAIL", mid.rel_rms, (double)kMidTol);
        if (trivial_pass && !mid_pass) {
            printf("[split] NOTE: carry plumbing is correct (trivial bit-exact) "
                   "but mid-cut exceeds the fp16-boundary budget — this is "
                   "drift/input-regime, not a carry bug.\n");
        }
        if (!trivial_pass) {
            printf("[split] NOTE: trivial cut is NOT near-exact — carry "
                   "threading bug; interior-cut error is meaningless until "
                   "this is fixed.\n");
        }
        printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
        rc = pass ? 0 : 1;
    } while (false);

    if (st_full) sd_split_state_free(st_full);
    free_sd_ctx(ctx);
    return rc;
}
