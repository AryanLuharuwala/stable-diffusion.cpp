#ifndef __DIFFUSION_MODEL_H__
#define __DIFFUSION_MODEL_H__

#include <optional>
#include "anima.hpp"
#include "ernie_image.hpp"
#include "flux.hpp"
#include "hidream_o1.hpp"
#include "ltxv.hpp"
#include "mmdit.hpp"
#include "qwen_image.hpp"
#include "tensor_ggml.hpp"
#include "unet.hpp"
#include "wan.hpp"
#include "z_image.hpp"

struct DiffusionParams {
    const sd::Tensor<float>* x                                         = nullptr;
    const sd::Tensor<float>* timesteps                                 = nullptr;
    const sd::Tensor<float>* audio_x                                   = nullptr;
    const sd::Tensor<float>* audio_timesteps                           = nullptr;
    const sd::Tensor<float>* context                                   = nullptr;
    const sd::Tensor<float>* c_concat                                  = nullptr;
    const sd::Tensor<float>* y                                         = nullptr;
    const sd::Tensor<int32_t>* t5_ids                                  = nullptr;
    const sd::Tensor<float>* t5_weights                                = nullptr;
    const sd::Tensor<float>* guidance                                  = nullptr;
    const std::vector<sd::Tensor<float>>* ref_latents                  = nullptr;
    const sd::Tensor<int32_t>* input_ids                               = nullptr;
    const sd::Tensor<int32_t>* input_pos                               = nullptr;
    const sd::Tensor<int32_t>* token_types                             = nullptr;
    const sd::Tensor<int32_t>* vinput_mask                             = nullptr;
    const std::vector<sd::Tensor<float>>* vlm_images                   = nullptr;
    const std::vector<std::pair<int, sd::Tensor<float>>>* image_embeds = nullptr;
    bool increase_ref_index                                            = false;
    int num_video_frames                                               = -1;
    const std::vector<sd::Tensor<float>>* controls                     = nullptr;
    float control_strength                                             = 0.f;
    const sd::Tensor<float>* vace_context                              = nullptr;
    float vace_strength                                                = 1.f;
    int audio_length                                                   = 0;
    float frame_rate                                                   = 24.f;
    const sd::Tensor<float>* video_positions                           = nullptr;
    const std::vector<int>* skip_layers                                = nullptr;
};

template <typename T>
static inline const sd::Tensor<T>& tensor_or_empty(const sd::Tensor<T>* tensor) {
    static const sd::Tensor<T> kEmpty;
    return tensor != nullptr ? *tensor : kEmpty;
}

// CF12-W6a: host-side carry-state extracted at the UNet middle-block boundary.
// Other diffusion backbones (DiT/MMDiT/Flux/Wan/...) do not currently expose a
// split surface — compute_half0/compute_half1 below return false on them.
struct DiffusionHalfCarry {
    sd::Tensor<float>              h;
    std::vector<sd::Tensor<float>> hs;
    sd::Tensor<float>              emb;
};

struct DiffusionModel {
    virtual std::string get_desc()                                               = 0;
    virtual sd::Tensor<float> compute(int n_threads,
                                      const DiffusionParams& diffusion_params)   = 0;
    virtual void alloc_params_buffer()                                           = 0;
    virtual void free_params_buffer()                                            = 0;
    virtual void free_compute_buffer()                                           = 0;
    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) = 0;
    virtual size_t get_params_buffer_size()                                      = 0;
    virtual void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter){};
    virtual int64_t get_adm_in_channels()                            = 0;
    virtual void set_flash_attention_enabled(bool enabled)           = 0;
    virtual void set_max_graph_vram_bytes(size_t max_vram_bytes)     = 0;
    virtual void set_circular_axes(bool circular_x, bool circular_y) = 0;

    // CF12-W6a: cross-rig block-split surface. Default returns false (not
    // supported) so non-UNet backbones don't need to implement it.
    virtual bool supports_block_split() const { return false; }
    virtual bool compute_half0(int n_threads,
                               const DiffusionParams& diffusion_params,
                               DiffusionHalfCarry& carry_out) {
        (void)n_threads; (void)diffusion_params; (void)carry_out;
        return false;
    }
    virtual sd::Tensor<float> compute_half1(int n_threads,
                                            const DiffusionParams& diffusion_params,
                                            const DiffusionHalfCarry& carry) {
        (void)n_threads; (void)diffusion_params; (void)carry;
        return {};
    }

    // CF12-W7: N-way generalization of the 2-way half split. split_block_count()
    // returns the number of linear blocks the backbone can be cut into (0 ==
    // not split-capable → caller falls back to whole-UNet / role-chain).
    // compute_range runs blocks [lo, hi): lo==0 seeds from diffusion_params
    // (x/timesteps/...); hi>=split_block_count() fills out_noise with the
    // final noise_pred; otherwise carry_out holds the {h, hs, emb} handed to
    // the next stage. carry_in is ignored when lo==0.
    virtual int split_block_count() const { return 0; }
    virtual bool compute_range(int n_threads,
                               int lo,
                               int hi,
                               const DiffusionHalfCarry& carry_in,
                               const DiffusionParams& diffusion_params,
                               DiffusionHalfCarry& carry_out,
                               sd::Tensor<float>& out_noise) {
        (void)n_threads; (void)lo; (void)hi; (void)carry_in;
        (void)diffusion_params; (void)carry_out; (void)out_noise;
        return false;
    }

    virtual ~DiffusionModel() = default;
};

struct UNetModel : public DiffusionModel {
    UNetModelRunner unet;

    UNetModel(ggml_backend_t backend,
              ggml_backend_t params_backend,
              const String2TensorStorage& tensor_storage_map = {},
              SDVersion version                              = VERSION_SD1)
        : unet(backend, params_backend, tensor_storage_map, "model.diffusion_model", version) {
    }

    std::string get_desc() override {
        return unet.get_desc();
    }

    void alloc_params_buffer() override {
        unet.alloc_params_buffer();
    }

    void free_params_buffer() override {
        unet.free_params_buffer();
    }

    void free_compute_buffer() override {
        unet.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        unet.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return unet.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        unet.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return unet.unet.adm_in_channels;
    }

    void set_flash_attention_enabled(bool enabled) {
        unet.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        unet.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        unet.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<sd::Tensor<float>> empty_controls;
        return unet.compute(n_threads,
                            *diffusion_params.x,
                            *diffusion_params.timesteps,
                            tensor_or_empty(diffusion_params.context),
                            tensor_or_empty(diffusion_params.c_concat),
                            tensor_or_empty(diffusion_params.y),
                            diffusion_params.num_video_frames,
                            diffusion_params.controls ? *diffusion_params.controls : empty_controls,
                            diffusion_params.control_strength);
    }

    bool supports_block_split() const override { return true; }

    bool compute_half0(int n_threads,
                       const DiffusionParams& diffusion_params,
                       DiffusionHalfCarry& carry_out) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<sd::Tensor<float>> empty_controls;
        UNetModelRunner::SplitCarry runner_carry;
        const bool ok = unet.compute_half0(n_threads,
                                           *diffusion_params.x,
                                           *diffusion_params.timesteps,
                                           tensor_or_empty(diffusion_params.context),
                                           tensor_or_empty(diffusion_params.c_concat),
                                           tensor_or_empty(diffusion_params.y),
                                           diffusion_params.num_video_frames,
                                           diffusion_params.controls ? *diffusion_params.controls : empty_controls,
                                           diffusion_params.control_strength,
                                           runner_carry);
        if (!ok) {
            return false;
        }
        carry_out.h   = std::move(runner_carry.h);
        carry_out.hs  = std::move(runner_carry.hs);
        carry_out.emb = std::move(runner_carry.emb);
        return true;
    }

    sd::Tensor<float> compute_half1(int n_threads,
                                    const DiffusionParams& diffusion_params,
                                    const DiffusionHalfCarry& carry) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        static const std::vector<sd::Tensor<float>> empty_controls;
        UNetModelRunner::SplitCarry runner_carry;
        runner_carry.h   = carry.h;
        runner_carry.hs  = carry.hs;
        runner_carry.emb = carry.emb;
        return unet.compute_half1(n_threads,
                                  runner_carry,
                                  tensor_or_empty(diffusion_params.context),
                                  diffusion_params.num_video_frames,
                                  diffusion_params.controls ? *diffusion_params.controls : empty_controls,
                                  diffusion_params.control_strength,
                                  diffusion_params.x->dim());
    }

    int split_block_count() const override {
        return unet.unet.num_split_blocks();
    }

    bool compute_range(int n_threads,
                       int lo,
                       int hi,
                       const DiffusionHalfCarry& carry_in,
                       const DiffusionParams& diffusion_params,
                       DiffusionHalfCarry& carry_out,
                       sd::Tensor<float>& out_noise) override {
        static const std::vector<sd::Tensor<float>> empty_controls;
        UNetModelRunner::SplitCarry cin;
        cin.h   = carry_in.h;
        cin.hs  = carry_in.hs;
        cin.emb = carry_in.emb;

        // The noise_pred restores to the latent (x) dim. lo==0 has x; later
        // stages reconstruct it from the carried hidden state's rank.
        size_t dim_hint = 4;
        if (diffusion_params.x != nullptr) {
            dim_hint = diffusion_params.x->dim();
        } else if (!carry_in.h.empty()) {
            dim_hint = carry_in.h.dim();
        }

        UNetModelRunner::SplitCarry cout;
        const bool ok = unet.compute_range(n_threads, lo, hi, cin,
                                           tensor_or_empty(diffusion_params.x),
                                           tensor_or_empty(diffusion_params.timesteps),
                                           tensor_or_empty(diffusion_params.context),
                                           tensor_or_empty(diffusion_params.c_concat),
                                           tensor_or_empty(diffusion_params.y),
                                           diffusion_params.num_video_frames,
                                           diffusion_params.controls ? *diffusion_params.controls : empty_controls,
                                           diffusion_params.control_strength,
                                           dim_hint, cout, out_noise);
        if (!ok) {
            return false;
        }
        carry_out.h   = std::move(cout.h);
        carry_out.hs  = std::move(cout.hs);
        carry_out.emb = std::move(cout.emb);
        return true;
    }
};

// CF12-W7: a DiffusionModel whose per-step UNet eval is delegated to a remote
// callback (the coordinator drives the N-way block chain across rigs). The
// host runs sd.cpp's real sample() loop — denoiser scalings, sigmas, sampler
// all local and unchanged — and only the diffusion_model->compute(x_in, t,
// cond) call goes remote. Non-compute virtuals forward to `inner` (the real
// model, used for config like adm_in_channels) so setup paths still work.
using RemoteUNetFn = std::function<bool(const DiffusionParams& params,
                                        sd::Tensor<float>& out_eps)>;

struct RemoteUNetModel : public DiffusionModel {
    std::shared_ptr<DiffusionModel> inner;
    RemoteUNetFn fn;

    RemoteUNetModel(std::shared_ptr<DiffusionModel> inner_, RemoteUNetFn fn_)
        : inner(std::move(inner_)), fn(std::move(fn_)) {}

    std::string get_desc() override { return "remote-unet"; }
    void alloc_params_buffer() override {}
    void free_params_buffer() override {}
    void free_compute_buffer() override {}
    void get_param_tensors(std::map<std::string, ggml_tensor*>&) override {}
    size_t get_params_buffer_size() override { return 0; }
    int64_t get_adm_in_channels() override { return inner ? inner->get_adm_in_channels() : 0; }
    void set_flash_attention_enabled(bool) override {}
    void set_max_graph_vram_bytes(size_t) override {}
    void set_circular_axes(bool, bool) override {}

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        (void)n_threads;
        sd::Tensor<float> out;
        if (!fn || !fn(diffusion_params, out)) {
            return {};
        }
        return out;
    }
};

struct MMDiTModel : public DiffusionModel {
    MMDiTRunner mmdit;

    MMDiTModel(ggml_backend_t backend,
               ggml_backend_t params_backend,
               const String2TensorStorage& tensor_storage_map = {})
        : mmdit(backend, params_backend, tensor_storage_map, "model.diffusion_model") {
    }

    std::string get_desc() override {
        return mmdit.get_desc();
    }

    void alloc_params_buffer() override {
        mmdit.alloc_params_buffer();
    }

    void free_params_buffer() override {
        mmdit.free_params_buffer();
    }

    void free_compute_buffer() override {
        mmdit.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        mmdit.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return mmdit.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        mmdit.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768 + 1280;
    }

    void set_flash_attention_enabled(bool enabled) {
        mmdit.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        mmdit.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        mmdit.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<int> empty_skip_layers;
        return mmdit.compute(n_threads,
                             *diffusion_params.x,
                             *diffusion_params.timesteps,
                             tensor_or_empty(diffusion_params.context),
                             tensor_or_empty(diffusion_params.y),
                             diffusion_params.skip_layers ? *diffusion_params.skip_layers : empty_skip_layers);
    }
};

struct FluxModel : public DiffusionModel {
    Flux::FluxRunner flux;

    FluxModel(ggml_backend_t backend,
              ggml_backend_t params_backend,
              const String2TensorStorage& tensor_storage_map = {},
              SDVersion version                              = VERSION_FLUX,
              bool use_mask                                  = false)
        : flux(backend, params_backend, tensor_storage_map, "model.diffusion_model", version, use_mask) {
    }

    std::string get_desc() override {
        return flux.get_desc();
    }

    void alloc_params_buffer() override {
        flux.alloc_params_buffer();
    }

    void free_params_buffer() override {
        flux.free_params_buffer();
    }

    void free_compute_buffer() override {
        flux.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        flux.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return flux.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        flux.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        flux.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        flux.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        flux.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<sd::Tensor<float>> empty_ref_latents;
        static const std::vector<int> empty_skip_layers;
        return flux.compute(n_threads,
                            *diffusion_params.x,
                            *diffusion_params.timesteps,
                            tensor_or_empty(diffusion_params.context),
                            tensor_or_empty(diffusion_params.c_concat),
                            tensor_or_empty(diffusion_params.y),
                            tensor_or_empty(diffusion_params.guidance),
                            diffusion_params.ref_latents ? *diffusion_params.ref_latents : empty_ref_latents,
                            diffusion_params.increase_ref_index,
                            diffusion_params.skip_layers ? *diffusion_params.skip_layers : empty_skip_layers);
    }
};

struct AnimaModel : public DiffusionModel {
    std::string prefix;
    Anima::AnimaRunner anima;

    AnimaModel(ggml_backend_t backend,
               ggml_backend_t params_backend,
               const String2TensorStorage& tensor_storage_map = {},
               const std::string prefix                       = "model.diffusion_model")
        : prefix(prefix), anima(backend, params_backend, tensor_storage_map, prefix) {
    }

    std::string get_desc() override {
        return anima.get_desc();
    }

    void alloc_params_buffer() override {
        anima.alloc_params_buffer();
    }

    void free_params_buffer() override {
        anima.free_params_buffer();
    }

    void free_compute_buffer() override {
        anima.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        anima.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return anima.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        anima.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        anima.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        anima.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        anima.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        return anima.compute(n_threads,
                             *diffusion_params.x,
                             *diffusion_params.timesteps,
                             tensor_or_empty(diffusion_params.context),
                             tensor_or_empty(diffusion_params.t5_ids),
                             tensor_or_empty(diffusion_params.t5_weights));
    }
};

struct WanModel : public DiffusionModel {
    std::string prefix;
    WAN::WanRunner wan;

    WanModel(ggml_backend_t backend,
             ggml_backend_t params_backend,
             const String2TensorStorage& tensor_storage_map = {},
             const std::string prefix                       = "model.diffusion_model",
             SDVersion version                              = VERSION_WAN2)
        : prefix(prefix), wan(backend, params_backend, tensor_storage_map, prefix, version) {
    }

    std::string get_desc() override {
        return wan.get_desc();
    }

    void alloc_params_buffer() override {
        wan.alloc_params_buffer();
    }

    void free_params_buffer() override {
        wan.free_params_buffer();
    }

    void free_compute_buffer() override {
        wan.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        wan.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return wan.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        wan.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        wan.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        wan.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        wan.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        return wan.compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           tensor_or_empty(diffusion_params.context),
                           tensor_or_empty(diffusion_params.y),
                           tensor_or_empty(diffusion_params.c_concat),
                           sd::Tensor<float>(),
                           tensor_or_empty(diffusion_params.vace_context),
                           diffusion_params.vace_strength);
    }
};

struct QwenImageModel : public DiffusionModel {
    std::string prefix;
    Qwen::QwenImageRunner qwen_image;

    QwenImageModel(ggml_backend_t backend,
                   ggml_backend_t params_backend,
                   const String2TensorStorage& tensor_storage_map = {},
                   const std::string prefix                       = "model.diffusion_model",
                   SDVersion version                              = VERSION_QWEN_IMAGE,
                   bool zero_cond_t                               = false)
        : prefix(prefix), qwen_image(backend, params_backend, tensor_storage_map, prefix, version, zero_cond_t) {
    }

    std::string get_desc() override {
        return qwen_image.get_desc();
    }

    void alloc_params_buffer() override {
        qwen_image.alloc_params_buffer();
    }

    void free_params_buffer() override {
        qwen_image.free_params_buffer();
    }

    void free_compute_buffer() override {
        qwen_image.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        qwen_image.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return qwen_image.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        qwen_image.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        qwen_image.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        qwen_image.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        qwen_image.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<sd::Tensor<float>> empty_ref_latents;
        return qwen_image.compute(n_threads,
                                  *diffusion_params.x,
                                  *diffusion_params.timesteps,
                                  tensor_or_empty(diffusion_params.context),
                                  diffusion_params.ref_latents ? *diffusion_params.ref_latents : empty_ref_latents,
                                  true);
    }
};

struct HiDreamO1Model : public DiffusionModel {
    std::string prefix;
    HiDreamO1::HiDreamO1Runner hidream_o1;

    HiDreamO1Model(ggml_backend_t backend,
                   ggml_backend_t params_backend,
                   const String2TensorStorage& tensor_storage_map = {},
                   const std::string& prefix                      = "model")
        : prefix(prefix), hidream_o1(backend, params_backend, tensor_storage_map, prefix) {
    }

    std::string get_desc() override {
        return hidream_o1.get_desc();
    }

    void alloc_params_buffer() override {
        hidream_o1.alloc_params_buffer();
    }

    void free_params_buffer() override {
        hidream_o1.free_params_buffer();
    }

    void free_compute_buffer() override {
        hidream_o1.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        hidream_o1.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return hidream_o1.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        hidream_o1.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 0;
    }

    void set_flash_attention_enabled(bool enabled) {
        hidream_o1.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        hidream_o1.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        hidream_o1.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        GGML_ASSERT(diffusion_params.input_ids != nullptr);
        GGML_ASSERT(diffusion_params.input_pos != nullptr);
        GGML_ASSERT(diffusion_params.token_types != nullptr);
        static const std::vector<sd::Tensor<float>> empty_images;
        static const std::vector<std::pair<int, sd::Tensor<float>>> empty_image_embeds;
        return hidream_o1.compute(n_threads,
                                  *diffusion_params.x,
                                  *diffusion_params.timesteps,
                                  *diffusion_params.input_ids,
                                  *diffusion_params.input_pos,
                                  *diffusion_params.token_types,
                                  tensor_or_empty(diffusion_params.vinput_mask),
                                  diffusion_params.image_embeds ? *diffusion_params.image_embeds : empty_image_embeds,
                                  diffusion_params.ref_latents ? *diffusion_params.ref_latents : empty_images);
    }
};

struct ZImageModel : public DiffusionModel {
    std::string prefix;
    ZImage::ZImageRunner z_image;

    ZImageModel(ggml_backend_t backend,
                ggml_backend_t params_backend,
                const String2TensorStorage& tensor_storage_map = {},
                const std::string prefix                       = "model.diffusion_model",
                SDVersion version                              = VERSION_Z_IMAGE)
        : prefix(prefix), z_image(backend, params_backend, tensor_storage_map, prefix, version) {
    }

    std::string get_desc() override {
        return z_image.get_desc();
    }

    void alloc_params_buffer() override {
        z_image.alloc_params_buffer();
    }

    void free_params_buffer() override {
        z_image.free_params_buffer();
    }

    void free_compute_buffer() override {
        z_image.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        z_image.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return z_image.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        z_image.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        z_image.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        z_image.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        z_image.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<sd::Tensor<float>> empty_ref_latents;
        return z_image.compute(n_threads,
                               *diffusion_params.x,
                               *diffusion_params.timesteps,
                               tensor_or_empty(diffusion_params.context),
                               diffusion_params.ref_latents ? *diffusion_params.ref_latents : empty_ref_latents,
                               true);
    }
};

struct ErnieImageModel : public DiffusionModel {
    std::string prefix;
    ErnieImage::ErnieImageRunner ernie_image;

    ErnieImageModel(ggml_backend_t backend,
                    ggml_backend_t params_backend,
                    const String2TensorStorage& tensor_storage_map = {},
                    const std::string prefix                       = "model.diffusion_model")
        : prefix(prefix), ernie_image(backend, params_backend, tensor_storage_map, prefix) {
    }

    std::string get_desc() override {
        return ernie_image.get_desc();
    }

    void alloc_params_buffer() override {
        ernie_image.alloc_params_buffer();
    }

    void free_params_buffer() override {
        ernie_image.free_params_buffer();
    }

    void free_compute_buffer() override {
        ernie_image.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        ernie_image.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return ernie_image.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        ernie_image.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        ernie_image.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        ernie_image.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        ernie_image.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        return ernie_image.compute(n_threads,
                                   *diffusion_params.x,
                                   *diffusion_params.timesteps,
                                   tensor_or_empty(diffusion_params.context));
    }
};

struct LTXAVModel : public DiffusionModel {
    std::string prefix;
    LTXV::LTXAVRunner ltxav;

    LTXAVModel(ggml_backend_t backend,
               ggml_backend_t params_backend,
               const String2TensorStorage& tensor_storage_map = {},
               const std::string prefix                       = "model.diffusion_model")
        : prefix(prefix), ltxav(backend, params_backend, tensor_storage_map, prefix) {
    }

    std::string get_desc() override {
        return ltxav.get_desc();
    }

    void alloc_params_buffer() override {
        ltxav.alloc_params_buffer();
    }

    void free_params_buffer() override {
        ltxav.free_params_buffer();
    }

    void free_compute_buffer() override {
        ltxav.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        ltxav.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return ltxav.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        ltxav.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 0;
    }

    void set_flash_attention_enabled(bool enabled) override {
        ltxav.set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        ltxav.set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        ltxav.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        return ltxav.compute(n_threads,
                             *diffusion_params.x,
                             *diffusion_params.timesteps,
                             tensor_or_empty(diffusion_params.context),
                             tensor_or_empty(diffusion_params.audio_x),
                             tensor_or_empty(diffusion_params.audio_timesteps),
                             diffusion_params.audio_length,
                             diffusion_params.frame_rate,
                             tensor_or_empty(diffusion_params.video_positions));
    }
};

#endif
