#ifndef __STABLE_DIFFUSION_H__
#define __STABLE_DIFFUSION_H__

#if defined(_WIN32) || defined(__CYGWIN__)
#ifndef SD_BUILD_SHARED_LIB
#define SD_API
#else
#ifdef SD_BUILD_DLL
#define SD_API __declspec(dllexport)
#else
#define SD_API __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define SD_API __attribute__((visibility("default")))
#else
#define SD_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum rng_type_t {
    STD_DEFAULT_RNG,
    CUDA_RNG,
    CPU_RNG,
    RNG_TYPE_COUNT
};

enum sample_method_t {
    EULER_SAMPLE_METHOD,
    EULER_A_SAMPLE_METHOD,
    HEUN_SAMPLE_METHOD,
    DPM2_SAMPLE_METHOD,
    DPMPP2S_A_SAMPLE_METHOD,
    DPMPP2M_SAMPLE_METHOD,
    DPMPP2Mv2_SAMPLE_METHOD,
    IPNDM_SAMPLE_METHOD,
    IPNDM_V_SAMPLE_METHOD,
    LCM_SAMPLE_METHOD,
    DDIM_TRAILING_SAMPLE_METHOD,
    TCD_SAMPLE_METHOD,
    RES_MULTISTEP_SAMPLE_METHOD,
    RES_2S_SAMPLE_METHOD,
    ER_SDE_SAMPLE_METHOD,
    EULER_CFG_PP_SAMPLE_METHOD,
    EULER_A_CFG_PP_SAMPLE_METHOD,
    EULER_GE_SAMPLE_METHOD,
    SAMPLE_METHOD_COUNT
};

enum scheduler_t {
    DISCRETE_SCHEDULER,
    KARRAS_SCHEDULER,
    EXPONENTIAL_SCHEDULER,
    AYS_SCHEDULER,
    GITS_SCHEDULER,
    SGM_UNIFORM_SCHEDULER,
    SIMPLE_SCHEDULER,
    SMOOTHSTEP_SCHEDULER,
    KL_OPTIMAL_SCHEDULER,
    LCM_SCHEDULER,
    BONG_TANGENT_SCHEDULER,
    LTX2_SCHEDULER,
    SCHEDULER_COUNT
};

enum prediction_t {
    EPS_PRED,
    V_PRED,
    EDM_V_PRED,
    FLOW_PRED,
    FLUX_FLOW_PRED,
    FLUX2_FLOW_PRED,
    PREDICTION_COUNT
};

// same as enum ggml_type
enum sd_type_t {
    SD_TYPE_F32  = 0,
    SD_TYPE_F16  = 1,
    SD_TYPE_Q4_0 = 2,
    SD_TYPE_Q4_1 = 3,
    // SD_TYPE_Q4_2 = 4, support has been removed
    // SD_TYPE_Q4_3 = 5, support has been removed
    SD_TYPE_Q5_0    = 6,
    SD_TYPE_Q5_1    = 7,
    SD_TYPE_Q8_0    = 8,
    SD_TYPE_Q8_1    = 9,
    SD_TYPE_Q2_K    = 10,
    SD_TYPE_Q3_K    = 11,
    SD_TYPE_Q4_K    = 12,
    SD_TYPE_Q5_K    = 13,
    SD_TYPE_Q6_K    = 14,
    SD_TYPE_Q8_K    = 15,
    SD_TYPE_IQ2_XXS = 16,
    SD_TYPE_IQ2_XS  = 17,
    SD_TYPE_IQ3_XXS = 18,
    SD_TYPE_IQ1_S   = 19,
    SD_TYPE_IQ4_NL  = 20,
    SD_TYPE_IQ3_S   = 21,
    SD_TYPE_IQ2_S   = 22,
    SD_TYPE_IQ4_XS  = 23,
    SD_TYPE_I8      = 24,
    SD_TYPE_I16     = 25,
    SD_TYPE_I32     = 26,
    SD_TYPE_I64     = 27,
    SD_TYPE_F64     = 28,
    SD_TYPE_IQ1_M   = 29,
    SD_TYPE_BF16    = 30,
    // SD_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // SD_TYPE_Q4_0_4_8 = 32,
    // SD_TYPE_Q4_0_8_8 = 33,
    SD_TYPE_TQ1_0 = 34,
    SD_TYPE_TQ2_0 = 35,
    // SD_TYPE_IQ4_NL_4_4 = 36,
    // SD_TYPE_IQ4_NL_4_8 = 37,
    // SD_TYPE_IQ4_NL_8_8 = 38,
    SD_TYPE_MXFP4 = 39,  // MXFP4 (1 block)
    SD_TYPE_NVFP4 = 40,  // NVFP4 (4 blocks, E4M3 scale)
    SD_TYPE_Q1_0  = 41,
    SD_TYPE_COUNT = 42,
};

enum sd_log_level_t {
    SD_LOG_DEBUG,
    SD_LOG_INFO,
    SD_LOG_WARN,
    SD_LOG_ERROR
};

enum preview_t {
    PREVIEW_NONE,
    PREVIEW_PROJ,
    PREVIEW_TAE,
    PREVIEW_VAE,
    PREVIEW_COUNT
};

enum lora_apply_mode_t {
    LORA_APPLY_AUTO,
    LORA_APPLY_IMMEDIATELY,
    LORA_APPLY_AT_RUNTIME,
    LORA_APPLY_MODE_COUNT,
};

typedef struct {
    bool enabled;
    bool temporal_tiling;
    int tile_size_x;
    int tile_size_y;
    float target_overlap;
    float rel_size_x;
    float rel_size_y;
    const char* extra_tiling_args;
} sd_tiling_params_t;

typedef struct {
    const char* name;
    const char* path;
} sd_embedding_t;

typedef struct {
    const char* model_path;
    const char* clip_l_path;
    const char* clip_g_path;
    const char* clip_vision_path;
    const char* t5xxl_path;
    const char* llm_path;
    const char* llm_vision_path;
    const char* diffusion_model_path;
    const char* high_noise_diffusion_model_path;
    const char* embeddings_connectors_path;
    const char* vae_path;
    const char* audio_vae_path;
    const char* taesd_path;
    const char* control_net_path;
    const sd_embedding_t* embeddings;
    uint32_t embedding_count;
    const char* photo_maker_path;
    const char* tensor_type_rules;
    bool vae_decode_only;
    bool free_params_immediately;
    int n_threads;
    enum sd_type_t wtype;
    enum rng_type_t rng_type;
    enum rng_type_t sampler_rng_type;
    enum prediction_t prediction;
    enum lora_apply_mode_t lora_apply_mode;
    bool offload_params_to_cpu;
    bool enable_mmap;
    bool keep_clip_on_cpu;
    bool keep_control_net_on_cpu;
    bool keep_vae_on_cpu;
    bool flash_attn;
    bool diffusion_flash_attn;
    bool tae_preview_only;
    bool diffusion_conv_direct;
    bool vae_conv_direct;
    bool circular_x;
    bool circular_y;
    bool force_sdxl_vae_conv_scale;
    bool chroma_use_dit_mask;
    bool chroma_use_t5_mask;
    int chroma_t5_mask_pad;
    bool qwen_image_zero_cond_t;
    float max_vram;  // GiB budget for graph-cut segmented param offload (0 = disabled, -1 = auto free VRAM minus 1 GiB)
    const char* backend;
    const char* params_backend;
} sd_ctx_params_t;

typedef struct {
    uint32_t sample_rate;
    uint32_t channels;
    uint64_t sample_count;
    float* data;
} sd_audio_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} sd_image_t;

typedef struct {
    int* layers;
    size_t layer_count;
    float layer_start;
    float layer_end;
    float scale;
} sd_slg_params_t;

typedef struct {
    float txt_cfg;
    float img_cfg;
    float distilled_guidance;
    sd_slg_params_t slg;
} sd_guidance_params_t;

typedef struct {
    sd_guidance_params_t guidance;
    enum scheduler_t scheduler;
    enum sample_method_t sample_method;
    int sample_steps;
    float eta;
    int shifted_timestep;
    float* custom_sigmas;
    int custom_sigmas_count;
    float flow_shift;
    const char* extra_sample_args;
} sd_sample_params_t;

typedef struct {
    sd_image_t* id_images;
    int id_images_count;
    const char* id_embed_path;
    float style_strength;
} sd_pm_params_t;  // photo maker

enum sd_cache_mode_t {
    SD_CACHE_DISABLED = 0,
    SD_CACHE_EASYCACHE,
    SD_CACHE_UCACHE,
    SD_CACHE_DBCACHE,
    SD_CACHE_TAYLORSEER,
    SD_CACHE_CACHE_DIT,
    SD_CACHE_SPECTRUM,
};

typedef struct {
    enum sd_cache_mode_t mode;
    float reuse_threshold;
    float start_percent;
    float end_percent;
    float error_decay_rate;
    bool use_relative_threshold;
    bool reset_error_on_compute;
    int Fn_compute_blocks;
    int Bn_compute_blocks;
    float residual_diff_threshold;
    int max_warmup_steps;
    int max_cached_steps;
    int max_continuous_cached_steps;
    int taylorseer_n_derivatives;
    int taylorseer_skip_interval;
    const char* scm_mask;
    bool scm_policy_dynamic;
    float spectrum_w;
    int spectrum_m;
    float spectrum_lam;
    int spectrum_window_size;
    float spectrum_flex_window;
    int spectrum_warmup_steps;
    float spectrum_stop_percent;
} sd_cache_params_t;

typedef struct {
    bool is_high_noise;
    float multiplier;
    const char* path;
} sd_lora_t;

enum sd_hires_upscaler_t {
    SD_HIRES_UPSCALER_NONE,
    SD_HIRES_UPSCALER_LATENT,
    SD_HIRES_UPSCALER_LATENT_NEAREST,
    SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT,
    SD_HIRES_UPSCALER_LATENT_ANTIALIASED,
    SD_HIRES_UPSCALER_LATENT_BICUBIC,
    SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED,
    SD_HIRES_UPSCALER_LANCZOS,
    SD_HIRES_UPSCALER_NEAREST,
    SD_HIRES_UPSCALER_MODEL,
    SD_HIRES_UPSCALER_COUNT,
};

typedef struct {
    bool enabled;
    enum sd_hires_upscaler_t upscaler;
    const char* model_path;
    float scale;
    int target_width;
    int target_height;
    int steps;
    float denoising_strength;
    int upscale_tile_size;
    float* custom_sigmas;
    int custom_sigmas_count;
} sd_hires_params_t;

typedef struct {
    const sd_lora_t* loras;
    uint32_t lora_count;
    const char* prompt;
    const char* negative_prompt;
    int clip_skip;
    sd_image_t init_image;
    sd_image_t* ref_images;
    int ref_images_count;
    bool auto_resize_ref_image;
    bool increase_ref_index;
    sd_image_t mask_image;
    int width;
    int height;
    sd_sample_params_t sample_params;
    float strength;
    int64_t seed;
    int batch_count;
    sd_image_t control_image;
    float control_strength;
    sd_pm_params_t pm_params;
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
    sd_hires_params_t hires;
} sd_img_gen_params_t;

typedef struct {
    const sd_lora_t* loras;
    uint32_t lora_count;
    const char* prompt;
    const char* negative_prompt;
    int clip_skip;
    sd_image_t init_image;
    sd_image_t end_image;
    sd_image_t* control_frames;
    int control_frames_size;
    int width;
    int height;
    sd_sample_params_t sample_params;
    sd_sample_params_t high_noise_sample_params;
    float moe_boundary;
    float strength;
    int64_t seed;
    int video_frames;
    int fps;
    float vace_strength;
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
    sd_hires_params_t hires;
} sd_vid_gen_params_t;

typedef struct sd_ctx_t sd_ctx_t;

typedef void (*sd_log_cb_t)(enum sd_log_level_t level, const char* text, void* data);
typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);
typedef void (*sd_preview_cb_t)(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data);

SD_API void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data);
SD_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data);
SD_API void sd_set_preview_callback(sd_preview_cb_t cb, enum preview_t mode, int interval, bool denoised, bool noisy, void* data);
SD_API int32_t sd_get_num_physical_cores();
SD_API const char* sd_get_system_info();
SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx);
SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* sd_ctx);

SD_API const char* sd_type_name(enum sd_type_t type);
SD_API enum sd_type_t str_to_sd_type(const char* str);
SD_API const char* sd_rng_type_name(enum rng_type_t rng_type);
SD_API enum rng_type_t str_to_rng_type(const char* str);
SD_API const char* sd_sample_method_name(enum sample_method_t sample_method);
SD_API enum sample_method_t str_to_sample_method(const char* str);
SD_API const char* sd_scheduler_name(enum scheduler_t scheduler);
SD_API enum scheduler_t str_to_scheduler(const char* str);
SD_API const char* sd_prediction_name(enum prediction_t prediction);
SD_API enum prediction_t str_to_prediction(const char* str);
SD_API const char* sd_preview_name(enum preview_t preview);
SD_API enum preview_t str_to_preview(const char* str);
SD_API const char* sd_lora_apply_mode_name(enum lora_apply_mode_t mode);
SD_API enum lora_apply_mode_t str_to_lora_apply_mode(const char* str);
SD_API const char* sd_hires_upscaler_name(enum sd_hires_upscaler_t upscaler);
SD_API enum sd_hires_upscaler_t str_to_sd_hires_upscaler(const char* str);

SD_API void sd_cache_params_init(sd_cache_params_t* cache_params);
SD_API void sd_hires_params_init(sd_hires_params_t* hires_params);

SD_API void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params);
SD_API char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params);

SD_API sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params);
SD_API void free_sd_ctx(sd_ctx_t* sd_ctx);
SD_API void free_sd_audio(sd_audio_t* audio);

SD_API void sd_sample_params_init(sd_sample_params_t* sample_params);
SD_API char* sd_sample_params_to_str(const sd_sample_params_t* sample_params);

SD_API enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx);
SD_API enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method);

SD_API void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params);
SD_API char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params);
SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params);

SD_API void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params);
SD_API bool generate_video(sd_ctx_t* sd_ctx,
                           const sd_vid_gen_params_t* sd_vid_gen_params,
                           sd_image_t** frames_out,
                           int* num_frames_out,
                           sd_audio_t** audio_out);

typedef struct upscaler_ctx_t upscaler_ctx_t;

SD_API upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path,
                                        bool offload_params_to_cpu,
                                        bool direct,
                                        int n_threads,
                                        int tile_size,
                                        const char* backend,
                                        const char* params_backend);
SD_API void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);

SD_API sd_image_t upscale(upscaler_ctx_t* upscaler_ctx,
                          sd_image_t input_image,
                          uint32_t upscale_factor);

SD_API int get_upscale_factor(upscaler_ctx_t* upscaler_ctx);

SD_API bool convert(const char* input_path,
                    const char* vae_path,
                    const char* output_path,
                    enum sd_type_t output_type,
                    const char* tensor_type_rules,
                    bool convert_name);

SD_API bool preprocess_canny(sd_image_t image,
                             float high_threshold,
                             float low_threshold,
                             float weak,
                             float strong,
                             bool inverse);

SD_API const char* sd_commit(void);
SD_API const char* sd_version(void);

// ─── Distributed per-layer UNet execution ─────────────────────────────────
//
// This surface lets an external scheduler run a single UNet forward pass as a
// chain of contiguous block ranges, optionally spread across several rigs, and
// drive the full denoise loop with a remote-UNet callback.  It is a strict
// extension: the math, carry-state {h, hs, emb} handling and fp16 boundaries
// are identical to the monolithic forward() — the split path is the same code
// walked in pieces, so a single whole-range call is bit-for-bit equivalent.
//
// See docs/distributed_unet.md for the end-to-end driving model.
//
// Block linearization.  forward() is a linear sequence of blocks:
//     [0]                     conv_in            (pushes hs[0])
//     [1 .. n_in-1]           input res / down   (each pushes a skip onto hs[])
//     [n_in]                  middle             (non-tiny backbones only)
//     [n_in+1 .. count-1]     output res / up    (each pops a skip off hs[])
// with the final out-conv folded onto the last stage.  Every per-block detail
// (downsample factor, layer names, attention placement, controlnet offsets) is
// a pure function of the block index, so a contiguous range [block_lo, block_hi)
// is fully reconstructible from the carry-state {h, hs, emb} — no extra fields.
//
// Block-range contract.  The range is half-open: [block_lo, block_hi) runs
// blocks block_lo .. block_hi-1.
//   * block_lo == 0           seeds the prelude (time/label emb + conv_in) from
//                             the staged x / timesteps / context / c_concat / y;
//                             no carry-in is read.
//   * block_lo  > 0           reads the carry {h, hs, emb} staged on the state.
//   * block_hi >= block_count appends the final out-conv and produces the
//                             noise-pred (sd_split_state_get_output).
//   * otherwise               produces a carry {h, hs, emb} for the next stage
//                             (sd_split_state_get_carry_tensor).
// Down-path skip residuals ride the hs[] stack forward across intermediate
// stages until the up-path pops them, so ANY tiling of [0, block_count) is
// valid and reassembles the exact whole-UNet result.
//
// Tensor layout.  All tensors crossing this surface are sd::Tensor shapes in
// WHCN order (ggml convention: dim 0 = width, 1 = height, 2 = channels,
// 3 = batch) — NOT NCHW.  Shapes are passed as int64 arrays whose element [0]
// is the fastest-varying (width) axis.  Data is plain contiguous fp32.

// Error codes returned by the split entry points and serialise helpers.
#define SD_SPLIT_OK         0
#define SD_SPLIT_EINVAL     1   // null/invalid argument or out-of-range block index
#define SD_SPLIT_ENOTSUP    2   // backbone is not split-capable (DiT/Flux/tiny-UNet)
#define SD_SPLIT_ESTATE     3   // carry-state mismatch (e.g. block_lo>0 without staged carry)
#define SD_SPLIT_EALLOC     4   // allocation failure

// ── Block introspection ───────────────────────────────────────────────────

// Number of linear UNet blocks the loaded model can be split into. SD1.x/SD2
// → 25, SDXL → 19 (conv_in + input + middle + output). Returns 0 for non-UNet
// backbones (DiT/MMDiT/Flux/...) and tiny-UNet variants, where the caller must
// fall back to the whole-UNet path. A planner partitions [0, count) across
// however many rigs are available (dynamic N).
SD_API int sd_unet_block_count(const sd_ctx_t* sd_ctx);

// Backbone tag derived from the loaded model — "sd1" | "sd2" | "sdxl" |
// "sd3" | "flux" | "pixart" | "unknown". Lets a scheduler advertise the real
// backbone rather than guessing from a filename.
SD_API const char* sd_loaded_backbone_tag(const sd_ctx_t* sd_ctx);

// ── Carry-state lifecycle ─────────────────────────────────────────────────
//
// The opaque sd_split_state_t holds, for one denoise step: the staged UNet
// inputs (x, timesteps, context, c_concat, y), the carry {h, hs[], emb} that
// flows between block ranges, the denoise cursor (step_idx / total_steps), and
// the final noise-pred. One state object is reused across the ranges of a step.

typedef struct sd_split_state_t sd_split_state_t;   // opaque

// Allocate / free the carry-state.
SD_API sd_split_state_t* sd_split_state_new(void);
SD_API void              sd_split_state_free(sd_split_state_t* state);

// Stage a UNet input for an upcoming block_lo==0 range. Shapes are WHCN (see
// "Tensor layout" above); pass `data=NULL, shape=NULL, ndims=0` for optional
// inputs (context, c_concat, y) the loaded backbone does not need. Data is
// COPIED into the state and may be freed by the caller after this returns.
SD_API int sd_split_state_set_input(
    sd_split_state_t* state,
    const char*       name,        // "x" | "timesteps" | "context" | "c_concat" | "y"
    const float*      data,
    const int64_t*    shape,
    int               ndims);

// Read the noise-pred produced by the most recent block_hi>=block_count range.
// Returns a borrowed pointer (valid until the next call that mutates the state);
// fills shape/ndims (WHCN). Pass non-NULL shape/ndims.
SD_API int sd_split_state_get_output(
    const sd_split_state_t* state,
    const float**           data,
    const int64_t**         shape,
    int*                    ndims);

// Read carry-state tensors h / emb / hs.i, e.g. to repack the carry for
// cross-rig transport. sd_split_state_get_carry_count returns the hs[] depth.
// Same borrow semantics as sd_split_state_get_output.
SD_API int sd_split_state_get_carry_count(const sd_split_state_t* state, int* hs_count);
SD_API int sd_split_state_get_carry_tensor(
    const sd_split_state_t* state,
    const char*             name,        // "h" | "emb" | "hs.0" .. "hs.N-1"
    const float**           data,
    const int64_t**         shape,
    int*                    ndims);

// Stage a carry-state tensor on a downstream rig before running a block_lo>0
// range. hs_count must be set first via sd_split_state_set_hs_count. Data is
// COPIED.
SD_API int sd_split_state_set_hs_count(sd_split_state_t* state, int hs_count);
SD_API int sd_split_state_set_carry_tensor(
    sd_split_state_t* state,
    const char*       name,
    const float*      data,
    const int64_t*    shape,
    int               ndims);

// Serialise / deserialise the carry-state into a self-describing raw blob
// (magic "SDSP" + version + step idx + hs count + length-prefixed fp32
// tensors). The returned buffer is owned by the library and must be freed with
// free(). In-process callers can use this directly; cross-rig callers typically
// wrap it in their own transport frame.
SD_API int sd_split_state_serialize  (const sd_split_state_t* state, uint8_t** out, size_t* nbytes);
SD_API int sd_split_state_deserialize(const uint8_t* in, size_t nbytes, sd_split_state_t** out);

// ── Block-range execution ─────────────────────────────────────────────────

// Run the linearized UNet blocks [block_lo, block_hi) for step_idx of a
// total_steps-long denoise schedule, following the block-range contract above.
// `state` carries the inputs in and the carry/noise-pred out. Returns
// SD_SPLIT_OK on success or one of the SD_SPLIT_E* codes (ENOTSUP when the
// backbone is not split-capable, ESTATE when a required carry was not staged).
SD_API int sd_compute_unet_split_range(
    sd_ctx_t*           sd_ctx,
    int                 block_lo,
    int                 block_hi,
    int                 step_idx,
    int                 total_steps,
    sd_split_state_t*   state);

// Legacy 2-way convenience over sd_compute_unet_split_range: which_half 0 runs
// input_blocks + middle (block_lo==0), which_half 1 runs output_blocks + final
// (block_hi==block_count). Equivalent to the N-way call with the cut at the
// middle-block boundary; retained for callers that only need the two halves.
SD_API int sd_compute_unet_split_step(
    sd_ctx_t*           sd_ctx,
    int                 which_half,
    int                 step_idx,
    int                 total_steps,
    sd_split_state_t*   state);

// ── Remote-denoise hook ────────────────────────────────────────────────────

// Per-step UNet delegate. When installed, the host runs sd.cpp's full sample()
// loop (text-encode, denoiser scalings, sigmas, sampler, VAE — all local and
// unchanged) but routes each per-step UNet eval through `cb`, which may drive
// the N-way block chain across rigs and return the eps.
//
// cb is invoked once per conditioning eval (cond and uncond separately under
// CFG). x is the scaled UNet input (WHCN), timestep the UNet t, ctx the
// c_crossattn (prompt embeds), y the c_vector (SDXL pooled; ndim 0 if absent).
// All shapes are WHCN. The callee mallocs *out_eps (the host frees it) and
// fills out_ne[0..*out_ndim) (out_ne points at a caller buffer of >= 8 slots).
// Return 0 on success, non-zero to abort the step.
typedef int (*sd_remote_unet_cb_t)(
    void*          user,
    const float*   x,   const int64_t* x_ne,   int x_ndim,
    float          timestep,
    const float*   ctx, const int64_t* ctx_ne, int ctx_ndim,
    const float*   y,   const int64_t* y_ne,   int y_ndim,
    float**        out_eps, int64_t* out_ne,   int* out_ndim);

// Install (or, with cb=NULL, clear) the remote-denoise delegate on sd_ctx.
SD_API void sd_set_remote_unet_cb(sd_ctx_t* sd_ctx, sd_remote_unet_cb_t cb, void* user);

// ─── CF12-W6b: TE / VAE bridges (llama-distributed extension) ──────────────
//
// Two narrow C surfaces over the existing internal `cond_stage_model->
// get_learned_condition` and `decode_first_stage` calls, so a TE-only or
// VAE-only worker can run those stages without going through the full
// `generate_image` orchestration.  The role bridge (sdcpp_roles.cpp) wraps
// the outputs in SDCD/SDT framing for cross-rig hop.

// Opaque conditioner result.  Holds cond.{crossattn,vector,concat} and the
// optional uncond.{...} tensors as host fp32 buffers.  Freed by sd_cond_free.
typedef struct sd_cond_t sd_cond_t;

SD_API sd_cond_t* sd_cond_new(void);
SD_API void       sd_cond_free(sd_cond_t* c);

// Run the loaded conditioner (CLIP-L / CLIP-G / T5 / …) on `prompt` and
// optionally `negative_prompt`.  Populates `out` with named tensors.  Pass
// negative_prompt==NULL or "" to skip the uncond half (no allocation).
// Returns SD_SPLIT_OK on success, SD_SPLIT_EINVAL on bad args, or
// SD_SPLIT_ENOTSUP if no cond_stage_model is loaded.
SD_API int sd_encode_condition(
    sd_ctx_t*       sd_ctx,
    const char*     prompt,
    const char*     negative_prompt,
    int             clip_skip,
    int             width,
    int             height,
    sd_cond_t*      out);

// True (1) when the result includes uncond.* tensors; 0 otherwise.
SD_API int sd_cond_has_uncond(const sd_cond_t* c);

// Fetch a named tensor by string.  Valid names:
//   "cond.crossattn"  | "cond.vector"  | "cond.concat"
//   "uncond.crossattn"| "uncond.vector"| "uncond.concat"
// The returned pointers are borrowed (lifetime tied to `c`); caller must
// not free them.  Returns SD_SPLIT_OK or SD_SPLIT_EINVAL.  Missing-but-
// expected tensors (e.g. c_vector on SD1.x) return EINVAL with shape=NULL.
SD_API int sd_cond_get_tensor(
    const sd_cond_t*  c,
    const char*       name,
    const float**     out_data,
    const int64_t**   out_shape,
    int*              out_ndims);

// VAE decode (latent → image).  `latent_data` is row-major fp32 in NCHW.
// On success writes `out_data` (malloc'd), `out_shape` (malloc'd), and
// `out_ndims` — caller frees with sd_vae_image_free.  Returns SD_SPLIT_OK
// or SD_SPLIT_EINVAL / SD_SPLIT_ENOTSUP.
SD_API int sd_decode_first_stage_to_floats(
    sd_ctx_t*         sd_ctx,
    const float*      latent_data,
    const int64_t*    latent_shape,
    int               latent_ndims,
    float**           out_data,
    int64_t**         out_shape,
    int*              out_ndims);

SD_API void sd_vae_image_free(float* data, int64_t* shape);

#ifdef __cplusplus
}
#endif

#endif  // __STABLE_DIFFUSION_H__
