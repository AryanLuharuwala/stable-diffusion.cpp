// Distributed per-layer UNet split-API correctness test (SD1.5).
//
// Proves the public "Distributed per-layer UNet execution" surface in
// include/stable-diffusion.h is correct by comparing, in-process:
//   A) a monolithic full-range eval   sd_compute_unet_split_range(0, count)
//   B) a chained 2-stage eval         [0, CUT) -> carry -> [CUT, count)
// for a single denoise step with deterministic inputs. The split path is
// documented to be the same code walked in pieces, so A and B must agree
// within the fp16-boundary error budget (~6%); a 2-way cut at the middle
// boundary is effectively bit-exact.
//
// This mirrors the proven reference validator
//   llama-distributed/python/dpp_runtime/validate_nway_split.py
// which builds x as a seeded ~N(0,1) latent of WHCN shape (64,64,4,1)
// (_encode_step_x_frame) at a fixed (step_idx, timestep) and chains the
// {h, hs, emb} carry between contiguous block ranges (_run_chain).
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

// 2-way cut at the legacy middle-block boundary: conv_in(0) + input(1..11) +
// middle(12) live below, output(13..24) above. Index 13 == "immediately after
// the middle block" (unet.hpp legacy cut point). Any contiguous cut is valid;
// this one is the natural / near-bit-exact one.
constexpr int kCut = 13;

constexpr float kRelTol = 0.12f;  // fp16-boundary budget (validator gates ~6%)

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

    // context: fixed deterministic crossattn embedding (zeroed-or-fixed; a
    // seeded fill keeps it non-degenerate without needing the text encoder).
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
bool thread_carry(const sd_split_state_t* src, sd_split_state_t* dst) {
    int hs_count = 0;
    if (sd_split_state_get_carry_count(src, &hs_count) != SD_SPLIT_OK) {
        fprintf(stderr, "get_carry_count failed\n");
        return false;
    }
    printf("[split] carry hs_count=%d\n", hs_count);
    if (sd_split_state_set_hs_count(dst, hs_count) != SD_SPLIT_OK) {
        fprintf(stderr, "set_hs_count failed\n");
        return false;
    }

    auto move_one = [&](const char* name) -> bool {
        const float*   data  = nullptr;
        const int64_t* shape = nullptr;
        int            ndims = 0;
        if (sd_split_state_get_carry_tensor(src, name, &data, &shape, &ndims) != SD_SPLIT_OK) {
            fprintf(stderr, "get_carry_tensor(%s) failed\n", name);
            return false;
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
    sd_split_state_t* st_a    = nullptr;
    sd_split_state_t* st_b    = nullptr;

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

        // ── A: monolithic full-range [0, count) ─────────────────────────────
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

        // ── B: chained 2-stage [0, CUT) -> carry -> [CUT, count) ────────────
        st_a = sd_split_state_new();
        if (st_a == nullptr || !stage_inputs(st_a)) break;
        r = sd_compute_unet_split_range(ctx, 0, kCut, /*step_idx=*/0,
                                        /*total_steps=*/1, st_a);
        if (r != SD_SPLIT_OK) {
            fprintf(stderr, "stage0 [0,%d) rc=%d\n", kCut, r);
            break;
        }

        st_b = sd_split_state_new();
        if (st_b == nullptr) break;
        // Downstream stage still needs the staged x/timesteps/context/y in
        // params (they ride the state alongside the carry).
        if (!stage_inputs(st_b)) break;
        if (!thread_carry(st_a, st_b)) break;
        r = sd_compute_unet_split_range(ctx, kCut, count, /*step_idx=*/0,
                                        /*total_steps=*/1, st_b);
        if (r != SD_SPLIT_OK) {
            fprintf(stderr, "stage1 [%d,%d) rc=%d\n", kCut, count, r);
            break;
        }
        std::vector<float> out_split;
        if (!copy_output(st_b, out_split)) break;
        printf("[split] B 2-stage     numel=%zu (cut=%d)\n", out_split.size(), kCut);

        // ── Compare ─────────────────────────────────────────────────────────
        if (out_full.size() != out_split.size() || out_full.empty()) {
            fprintf(stderr, "FAIL: numel mismatch A=%zu B=%zu\n",
                    out_full.size(), out_split.size());
            printf("RESULT: FAIL\n");
            break;
        }
        double max_abs = 0.0;
        double max_rel = 0.0;
        double sum_sq_diff = 0.0;
        double sum_sq_ref  = 0.0;
        for (size_t i = 0; i < out_full.size(); ++i) {
            const double a = out_full[i];
            const double b = out_split[i];
            const double d = std::fabs(a - b);
            if (d > max_abs) max_abs = d;
            const double denom = std::fabs(a) > 1e-6 ? std::fabs(a) : 1e-6;
            const double rel = d / denom;
            if (rel > max_rel) max_rel = rel;
            sum_sq_diff += d * d;
            sum_sq_ref  += a * a;
        }
        const double rel_rms =
            std::sqrt(sum_sq_diff / (sum_sq_ref > 1e-12 ? sum_sq_ref : 1e-12));

        printf("[split] max_abs_diff=%.6e  max_rel_diff=%.4f  rel_rms=%.4f (%.2f%%)\n",
               max_abs, max_rel, rel_rms, rel_rms * 100.0);

        // Gate on RMS relative error vs signal (matches the validator's `rel`
        // metric); also report per-element max_rel for diagnostics.
        const bool pass = rel_rms < kRelTol;
        printf("RESULT: %s (rel_rms=%.4f, tol=%.2f)\n",
               pass ? "PASS" : "FAIL", rel_rms, kRelTol);
        rc = pass ? 0 : 1;
    } while (false);

    if (st_b)    sd_split_state_free(st_b);
    if (st_a)    sd_split_state_free(st_a);
    if (st_full) sd_split_state_free(st_full);
    free_sd_ctx(ctx);
    return rc;
}
