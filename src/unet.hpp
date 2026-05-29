#ifndef __UNET_HPP__
#define __UNET_HPP__

#include "common_block.hpp"
#include "model.h"

/*==================================================== UnetModel =====================================================*/

#define UNET_GRAPH_SIZE 102400

class SpatialVideoTransformer : public SpatialTransformer {
protected:
    int64_t time_depth;
    int max_time_embed_period;

public:
    SpatialVideoTransformer(int64_t in_channels,
                            int64_t n_head,
                            int64_t d_head,
                            int64_t depth,
                            int64_t context_dim,
                            bool use_linear,
                            int64_t time_depth        = 1,
                            int max_time_embed_period = 10000)
        : SpatialTransformer(in_channels, n_head, d_head, depth, context_dim, use_linear),
          max_time_embed_period(max_time_embed_period) {
        // We will convert unet transformer linear to conv2d 1x1 when loading the weights, so use_linear is always False
        // use_spatial_context is always True
        // merge_strategy is always learned_with_images
        // merge_factor is loaded from weights
        // time_context_dim is always None
        // ff_in is always True
        // disable_self_attn is always False
        // disable_temporal_crossattention is always False

        int64_t inner_dim = n_head * d_head;

        GGML_ASSERT(depth == time_depth);
        GGML_ASSERT(in_channels == inner_dim);

        int64_t time_mix_d_head    = d_head;
        int64_t n_time_mix_heads   = n_head;
        int64_t time_mix_inner_dim = time_mix_d_head * n_time_mix_heads;  // equal to inner_dim
        int64_t time_context_dim   = context_dim;

        for (int i = 0; i < time_depth; i++) {
            std::string name = "time_stack." + std::to_string(i);
            blocks[name]     = std::shared_ptr<GGMLBlock>(new BasicTransformerBlock(inner_dim,
                                                                                    n_time_mix_heads,
                                                                                    time_mix_d_head,
                                                                                    time_context_dim,
                                                                                    true));
        }

        int64_t time_embed_dim     = in_channels * 4;
        blocks["time_pos_embed.0"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, time_embed_dim));
        // time_pos_embed.1 is nn.SiLU()
        blocks["time_pos_embed.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, in_channels));

        blocks["time_mixer"] = std::shared_ptr<GGMLBlock>(new AlphaBlender());
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* context,
                         int timesteps) {
        // x: [N, in_channels, h, w] aka [b*t, in_channels, h, w], t == timesteps
        // context: [N, max_position(aka n_context), hidden_size(aka context_dim)] aka [b*t, n_context, context_dim], t == timesteps
        // t_emb: [N, in_channels] aka [b*t, in_channels]
        // timesteps is num_frames
        // time_context is always None
        // image_only_indicator is always tensor([0.])
        // transformer_options is not used
        // GGML_ASSERT(ggml_n_dims(context) == 3);

        auto norm             = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm"]);
        auto proj_in          = std::dynamic_pointer_cast<Conv2d>(blocks["proj_in"]);
        auto proj_out         = std::dynamic_pointer_cast<Conv2d>(blocks["proj_out"]);
        auto time_pos_embed_0 = std::dynamic_pointer_cast<Linear>(blocks["time_pos_embed.0"]);
        auto time_pos_embed_2 = std::dynamic_pointer_cast<Linear>(blocks["time_pos_embed.2"]);
        auto time_mixer       = std::dynamic_pointer_cast<AlphaBlender>(blocks["time_mixer"]);

        auto x_in         = x;
        int64_t n         = x->ne[3];
        int64_t h         = x->ne[1];
        int64_t w         = x->ne[0];
        int64_t inner_dim = n_head * d_head;

        GGML_ASSERT(n == timesteps);  // We compute cond and uncond separately, so batch_size==1

        auto time_context    = context;  // [b*t, n_context, context_dim]
        auto spatial_context = context;
        // time_context_first_timestep = time_context[::timesteps]
        auto time_context_first_timestep = ggml_view_3d(ctx->ggml_ctx,
                                                        time_context,
                                                        time_context->ne[0],
                                                        time_context->ne[1],
                                                        1,
                                                        time_context->nb[1],
                                                        time_context->nb[2],
                                                        0);  // [b, n_context, context_dim]
        time_context                     = ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32,
                                                              time_context_first_timestep->ne[0],
                                                              time_context_first_timestep->ne[1],
                                                              time_context_first_timestep->ne[2] * h * w);
        time_context                     = ggml_repeat(ctx->ggml_ctx, time_context_first_timestep, time_context);  // [b*h*w, n_context, context_dim]

        x = norm->forward(ctx, x);
        x = proj_in->forward(ctx, x);  // [N, inner_dim, h, w]

        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));  // [N, h, w, inner_dim]
        x = ggml_reshape_3d(ctx->ggml_ctx, x, inner_dim, w * h, n);                // [N, h * w, inner_dim]

        auto num_frames = ggml_arange(ctx->ggml_ctx, 0.f, static_cast<float>(timesteps), 1.f);
        // since b is 1, no need to do repeat
        auto t_emb = ggml_ext_timestep_embedding(ctx->ggml_ctx, num_frames, static_cast<int>(in_channels), max_time_embed_period);  // [N, in_channels]

        auto emb = time_pos_embed_0->forward(ctx, t_emb);
        emb      = ggml_silu_inplace(ctx->ggml_ctx, emb);
        emb      = time_pos_embed_2->forward(ctx, emb);                             // [N, in_channels]
        emb      = ggml_reshape_3d(ctx->ggml_ctx, emb, emb->ne[0], 1, emb->ne[1]);  // [N, 1, in_channels]

        for (int i = 0; i < depth; i++) {
            std::string transformer_name = "transformer_blocks." + std::to_string(i);
            std::string time_stack_name  = "time_stack." + std::to_string(i);

            auto block     = std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[transformer_name]);
            auto mix_block = std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[time_stack_name]);

            x = block->forward(ctx, x, spatial_context);  // [N, h * w, inner_dim]

            // in_channels == inner_dim
            auto x_mix = x;
            x_mix      = ggml_add(ctx->ggml_ctx, x_mix, emb);  // [N, h * w, inner_dim]

            int64_t N = x_mix->ne[2];
            int64_t T = timesteps;
            int64_t B = N / T;
            int64_t S = x_mix->ne[1];
            int64_t C = x_mix->ne[0];

            x_mix = ggml_reshape_4d(ctx->ggml_ctx, x_mix, C, S, T, B);                         // (b t) s c -> b t s c
            x_mix = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x_mix, 0, 2, 1, 3));  // b t s c -> b s t c
            x_mix = ggml_reshape_3d(ctx->ggml_ctx, x_mix, C, T, S * B);                        // b s t c -> (b s) t c

            x_mix = mix_block->forward(ctx, x_mix, time_context);  // [B * h * w, T, inner_dim]

            x_mix = ggml_reshape_4d(ctx->ggml_ctx, x_mix, C, T, S, B);                         // (b s) t c -> b s t c
            x_mix = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x_mix, 0, 2, 1, 3));  // b s t c -> b t s c
            x_mix = ggml_reshape_3d(ctx->ggml_ctx, x_mix, C, S, T * B);                        // b t s c -> (b t) s c

            x = time_mixer->forward(ctx, x, x_mix);  // [N, h * w, inner_dim]
        }

        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [N, inner_dim, h * w]
        x = ggml_reshape_4d(ctx->ggml_ctx, x, w, h, inner_dim, n);                 // [N, inner_dim, h, w]

        // proj_out
        x = proj_out->forward(ctx, x);  // [N, in_channels, h, w]

        x = ggml_add(ctx->ggml_ctx, x, x_in);
        return x;
    }
};

// ldm.modules.diffusionmodules.openaimodel.UNetModel
class UnetModelBlock : public GGMLBlock {
protected:
    SDVersion version = VERSION_SD1;
    // network hparams
    int in_channels                        = 4;
    int out_channels                       = 4;
    int num_res_blocks                     = 2;
    std::vector<int> attention_resolutions = {4, 2, 1};
    std::vector<int> channel_mult          = {1, 2, 4, 4};
    std::vector<int> transformer_depth     = {1, 1, 1, 1};
    int time_embed_dim                     = 1280;  // model_channels*4
    int num_heads                          = 8;
    int num_head_channels                  = -1;   // channels // num_heads
    int context_dim                        = 768;  // 1024 for VERSION_SD2, 2048 for VERSION_SDXL
    bool use_linear_projection             = false;
    bool tiny_unet                         = false;

public:
    int model_channels  = 320;
    int adm_in_channels = 2816;  // only for VERSION_SDXL/SVD

    // CF12-W6a: ds factor at the middle-block boundary, derived from the
    // channel-mult ladder (each non-final level doubles ds). Used by the
    // split-runner's half1 graph to reconstruct the carry-state.
    int boundary_ds() const {
        int ds = 1;
        for (size_t i = 0; i + 1 < channel_mult.size(); ++i) {
            ds *= 2;
        }
        return ds;
    }

    // ─── CF12-W7: N-way UNet block-split schedule ────────────────────────
    //
    // The monolithic forward() is a linear sequence of blocks:
    //   [0]                       conv_in            (pushes hs[0])
    //   [1 .. n_in-1]             input res/down     (each pushes hs)
    //   [n_in]                    middle             (non-tiny only)
    //   [n_in+1 .. total-1]       output res/up      (each pops hs)
    // with a final out-conv folded onto the last stage (hi == total).
    //
    // `ds`, layer names, attention placement, upsample sublayer indices and
    // controlnet offsets are all pure functions of the block index, so any
    // contiguous range [lo, hi) is reconstructible from the carry-state
    // {h, hs, emb} alone — no extra wire fields. forward_range() walks this
    // schedule; forward_half0/half1/forward delegate to it (so the existing
    // 2-way path is byte-identical and acts as a regression guard).
    struct BlockStep {
        enum Kind { ConvIn, InputRes, InputDown, Middle, OutputRes } kind = ConvIn;
        int  idx           = 0;      // input_block_idx / output_block_idx for layer names
        int  ds            = 1;      // downsample factor during this block (attention test)
        bool has_attn      = false;  // attention sublayer present
        bool upsample      = false;  // OutputRes: an upsample follows
        int  up_sample_idx = 1;      // OutputRes: upsample sublayer name index
    };

    // Pure simulation of the input/middle/output loops — records per-block
    // metadata without building any graph ops. Mirrors forward_half0/half1.
    std::vector<BlockStep> build_split_schedule() const {
        std::vector<BlockStep> sch;
        const size_t len_mults = channel_mult.size();
        auto attn_at = [&](int d) {
            return std::find(attention_resolutions.begin(), attention_resolutions.end(), d) != attention_resolutions.end();
        };
        { BlockStep s; s.kind = BlockStep::ConvIn; s.idx = 0; s.ds = 1; sch.push_back(s); }
        int input_block_idx = 0;
        int ds              = 1;
        for (int i = 0; i < (int)len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                BlockStep s; s.kind = BlockStep::InputRes; s.idx = input_block_idx; s.ds = ds;
                s.has_attn = attn_at(ds);
                sch.push_back(s);
            }
            if (i != (int)len_mults - 1) {
                ds *= 2;
                input_block_idx += 1;
                BlockStep s; s.kind = BlockStep::InputDown; s.idx = input_block_idx; s.ds = ds;
                sch.push_back(s);
            }
        }
        { BlockStep s; s.kind = BlockStep::Middle; s.ds = ds;
          s.has_attn = (version != VERSION_SDXL_SSD1B && version != VERSION_SDXL_VEGA);
          sch.push_back(s); }
        int output_block_idx = 0;
        for (int i = (int)len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                BlockStep s; s.kind = BlockStep::OutputRes; s.idx = output_block_idx; s.ds = ds;
                s.has_attn      = attn_at(ds);
                s.up_sample_idx = s.has_attn ? 2 : 1;
                if (i > 0 && j == num_res_blocks) {
                    s.upsample = true;
                    ds /= 2;
                }
                sch.push_back(s);
                output_block_idx += 1;
            }
        }
        return sch;
    }

    // Total splittable blocks; 0 == not split-capable (tiny UNet variants use
    // a different index dance that we don't slice — they fall back to the
    // whole-UNet / role-chain path).
    int num_split_blocks() const {
        if (tiny_unet) return 0;
        return (int)build_split_schedule().size();
    }

    // The legacy 2-way cut point: index immediately after the middle block.
    int split_mid_cut() const {
        const auto sch = build_split_schedule();
        for (size_t i = 0; i < sch.size(); ++i) {
            if (sch[i].kind == BlockStep::Middle) return (int)i + 1;
        }
        return (int)sch.size() / 2;
    }

    struct UnetHalfState;  // defined below (carry-state used by forward_range)

    // Run linearized blocks [lo, hi). lo==0 computes the prelude (emb +
    // conv_in) from x/timesteps/c_concat/y. hi==num_split_blocks() appends
    // the final out-conv (io.h becomes the noise_pred). Otherwise io is the
    // carry handed to the next stage.
    void forward_range(GGMLRunnerContext* ctx,
                       UnetHalfState& io,
                       int lo,
                       int hi,
                       ggml_tensor* x,
                       ggml_tensor* timesteps,
                       ggml_tensor* context,
                       ggml_tensor* c_concat              = nullptr,
                       ggml_tensor* y                     = nullptr,
                       int num_video_frames               = -1,
                       std::vector<ggml_tensor*> controls = {},
                       float control_strength             = 0.f);

    UnetModelBlock(SDVersion version = VERSION_SD1, const String2TensorStorage& tensor_storage_map = {})
        : version(version) {
        if (sd_version_is_sd2(version)) {
            context_dim           = 1024;
            num_head_channels     = 64;
            num_heads             = -1;
            use_linear_projection = true;
        } else if (sd_version_is_sdxl(version)) {
            context_dim           = 2048;
            attention_resolutions = {4, 2};
            channel_mult          = {1, 2, 4};
            transformer_depth     = {1, 2, 10};
            num_head_channels     = 64;
            num_heads             = -1;
            use_linear_projection = true;
            if (version == VERSION_SDXL_VEGA) {
                transformer_depth = {1, 1, 2};
            }
        } else if (version == VERSION_SVD) {
            in_channels           = 8;
            out_channels          = 4;
            context_dim           = 1024;
            adm_in_channels       = 768;
            num_head_channels     = 64;
            num_heads             = -1;
            use_linear_projection = true;
        }
        if (sd_version_is_inpaint(version)) {
            in_channels = 9;
        } else if (sd_version_is_unet_edit(version)) {
            in_channels = 8;
        }
        if (version == VERSION_SD1_TINY_UNET || version == VERSION_SD2_TINY_UNET || version == VERSION_SDXS_512_DS || version == VERSION_SDXS_09) {
            num_res_blocks = 1;
            channel_mult   = {1, 2, 4};
            tiny_unet      = true;
            if (version == VERSION_SDXS_512_DS) {
                attention_resolutions = {4, 2};  // here just like SDXL
            }
        }

        // dims is always 2
        // use_temporal_attention is always True for SVD

        blocks["time_embed.0"] = std::shared_ptr<GGMLBlock>(new Linear(model_channels, time_embed_dim));
        // time_embed_1 is nn.SiLU()
        blocks["time_embed.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));

        if (sd_version_is_sdxl(version) || version == VERSION_SVD) {
            blocks["label_emb.0.0"] = std::shared_ptr<GGMLBlock>(new Linear(adm_in_channels, time_embed_dim));
            // label_emb_1 is nn.SiLU()
            blocks["label_emb.0.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));
        }

        // input_blocks
        blocks["input_blocks.0.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, model_channels, {3, 3}, {1, 1}, {1, 1}));

        std::vector<int> input_block_chans;
        input_block_chans.push_back(model_channels);
        int ch              = model_channels;
        int input_block_idx = 0;
        int ds              = 1;

        auto get_resblock = [&](int64_t channels, int64_t emb_channels, int64_t out_channels) -> ResBlock* {
            if (version == VERSION_SVD) {
                return new VideoResBlock(channels, emb_channels, out_channels);
            } else {
                return new ResBlock(channels, emb_channels, out_channels);
            }
        };

        auto get_attention_layer = [&](int64_t in_channels,
                                       int64_t n_head,
                                       int64_t d_head,
                                       int64_t depth,
                                       int64_t context_dim) -> SpatialTransformer* {
            if (version == VERSION_SVD) {
                return new SpatialVideoTransformer(in_channels, n_head, d_head, depth, context_dim, use_linear_projection);
            } else {
                if (version == VERSION_SDXS_09 && n_head == 5) {
                    n_head = 1;    // to carry a special case of sdxs_09 into CrossAttentionLayer,
                    d_head = 320;  // works as long the product remains equal (5*64 == 1*320)
                }
                return new SpatialTransformer(in_channels, n_head, d_head, depth, context_dim, use_linear_projection);
            }
        };

        size_t len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, mult * model_channels));

                ch = mult * model_channels;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    int td           = transformer_depth[i];
                    if (version == VERSION_SDXL_SSD1B) {
                        if (i == 2) {
                            td = 4;
                        }
                    }
                    blocks[name] = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                  n_head,
                                                                                  d_head,
                                                                                  td,
                                                                                  context_dim));
                }
                input_block_chans.push_back(ch);
                if (tiny_unet) {
                    input_block_idx++;
                }
            }
            if (i != len_mults - 1) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(new DownSampleBlock(ch, ch));

                input_block_chans.push_back(ch);
                ds *= 2;
            }
        }

        // middle blocks
        int n_head = num_heads;
        int d_head = ch / num_heads;
        if (num_head_channels != -1) {
            d_head = num_head_channels;
            n_head = ch / d_head;
        }
        if (!tiny_unet) {
            blocks["middle_block.0"] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, ch));
            if (version != VERSION_SDXL_SSD1B && version != VERSION_SDXL_VEGA) {
                blocks["middle_block.1"] = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                          n_head,
                                                                                          d_head,
                                                                                          transformer_depth[transformer_depth.size() - 1],
                                                                                          context_dim));
                blocks["middle_block.2"] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, ch));
            }
        }
        // output_blocks
        int output_block_idx = 0;
        for (int i = (int)len_mults - 1; i >= 0; i--) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks + 1; j++) {
                int ich = input_block_chans.back();
                input_block_chans.pop_back();

                std::string name = "output_blocks." + std::to_string(output_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(get_resblock(ch + ich, time_embed_dim, mult * model_channels));

                ch                = mult * model_channels;
                int up_sample_idx = 1;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + ".1";
                    int td           = transformer_depth[i];
                    if (version == VERSION_SDXL_SSD1B) {
                        if (i == 2 && (j == 0 || j == 1)) {
                            td = 4;
                        }
                        if (i == 1 && (j == 1 || j == 2)) {
                            td = 1;
                        }
                    }
                    blocks[name] = std::shared_ptr<GGMLBlock>(get_attention_layer(ch, n_head, d_head, td, context_dim));

                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) {
                    if (tiny_unet) {
                        output_block_idx++;
                        if (output_block_idx == 2) {
                            up_sample_idx = 1;
                        }
                    }
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + "." + std::to_string(up_sample_idx);
                    blocks[name]     = std::shared_ptr<GGMLBlock>(new UpSampleBlock(ch, ch));

                    ds /= 2;
                }

                output_block_idx += 1;
            }
        }

        // out
        blocks["out.0"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(ch));  // ch == model_channels
        // out_1 is nn.SiLU()
        blocks["out.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(model_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    ggml_tensor* resblock_forward(std::string name,
                                  GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* emb,
                                  int num_video_frames) {
        if (version == VERSION_SVD) {
            auto block = std::dynamic_pointer_cast<VideoResBlock>(blocks[name]);

            return block->forward(ctx, x, emb, num_video_frames);
        } else {
            auto block = std::dynamic_pointer_cast<ResBlock>(blocks[name]);

            return block->forward(ctx, x, emb);
        }
    }

    ggml_tensor* attention_layer_forward(std::string name,
                                         GGMLRunnerContext* ctx,
                                         ggml_tensor* x,
                                         ggml_tensor* context,
                                         int timesteps) {
        if (version == VERSION_SVD) {
            auto block = std::dynamic_pointer_cast<SpatialVideoTransformer>(blocks[name]);

            return block->forward(ctx, x, context, timesteps);
        } else {
            auto block = std::dynamic_pointer_cast<SpatialTransformer>(blocks[name]);

            return block->forward(ctx, x, context);
        }
    }

    // ─── CF12-W6a: UNet block-split carry-state ──────────────────────────
    //
    // forward_half0 runs prelude + input_blocks + middle_block.
    // forward_half1 runs output_blocks + final out conv.
    // When chained inline (same graph build), the result is byte-identical
    // to the monolithic forward() that lived here before — see forward()
    // below which simply wires the two halves back-to-back.
    //
    // The split exists so an external driver can build two graphs with
    // independent compute-buffer footprints (one rig owns the input-side
    // weights, another owns the output-side weights) and ferry the
    // carry-state between them.
    struct UnetHalfState {
        ggml_tensor* h   = nullptr;            // hidden state after middle_block
        std::vector<ggml_tensor*> hs;          // skip residual stack (LIFO)
        ggml_tensor* emb = nullptr;            // time + label embedding
        int ds           = 1;                  // downsample factor at boundary
    };

    UnetHalfState forward_half0(GGMLRunnerContext* ctx,
                                ggml_tensor* x,
                                ggml_tensor* timesteps,
                                ggml_tensor* context,
                                ggml_tensor* c_concat              = nullptr,
                                ggml_tensor* y                     = nullptr,
                                int num_video_frames               = -1,
                                std::vector<ggml_tensor*> controls = {},
                                float control_strength             = 0.f);

    ggml_tensor* forward_half1(GGMLRunnerContext* ctx,
                               const UnetHalfState& st,
                               ggml_tensor* context,
                               int num_video_frames               = -1,
                               std::vector<ggml_tensor*> controls = {},
                               float control_strength             = 0.f);

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* timesteps,
                         ggml_tensor* context,
                         ggml_tensor* c_concat              = nullptr,
                         ggml_tensor* y                     = nullptr,
                         int num_video_frames               = -1,
                         std::vector<ggml_tensor*> controls = {},
                         float control_strength             = 0.f) {
        // x: [N, in_channels, h, w] or [N, in_channels/2, h, w]
        // timesteps: [N,]
        // context: [N, max_position, hidden_size] or [1, max_position, hidden_size]. for example, [N, 77, 768]
        // c_concat: [N, in_channels, h, w] or [1, in_channels, h, w]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        // return: [N, out_channels, h, w]
        UnetHalfState st = forward_half0(ctx, x, timesteps, context, c_concat, y,
                                         num_video_frames, controls, control_strength);
        return forward_half1(ctx, st, context, num_video_frames, controls, control_strength);
    }
};

// ─── CF12-W6a: out-of-class definitions of forward_half0 / forward_half1 ──
// Defined here (still in the header, since UNetModelRunner is header-only)
// so the file stays a single translation-unit drop-in.

inline UnetModelBlock::UnetHalfState UnetModelBlock::forward_half0(
    GGMLRunnerContext* ctx,
    ggml_tensor* x,
    ggml_tensor* timesteps,
    ggml_tensor* context,
    ggml_tensor* c_concat,
    ggml_tensor* y,
    int num_video_frames,
    std::vector<ggml_tensor*> controls,
    float control_strength) {
    if (context != nullptr) {
        if (context->ne[2] != x->ne[3]) {
            context = ggml_repeat(ctx->ggml_ctx, context, ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32, context->ne[0], context->ne[1], x->ne[3]));
        }
    }

    if (c_concat != nullptr) {
        if (c_concat->ne[3] != x->ne[3]) {
            c_concat = ggml_repeat(ctx->ggml_ctx, c_concat, x);
        }
        x = ggml_concat(ctx->ggml_ctx, x, c_concat, 2);
    }

    if (y != nullptr) {
        if (y->ne[1] != x->ne[3]) {
            y = ggml_repeat(ctx->ggml_ctx, y, ggml_new_tensor_2d(ctx->ggml_ctx, GGML_TYPE_F32, y->ne[0], x->ne[3]));
        }
    }

    auto time_embed_0     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.0"]);
    auto time_embed_2     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.2"]);
    auto input_blocks_0_0 = std::dynamic_pointer_cast<Conv2d>(blocks["input_blocks.0.0"]);

        auto t_emb = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, model_channels);  // [N, model_channels]

        auto emb = time_embed_0->forward(ctx, t_emb);
        emb      = ggml_silu_inplace(ctx->ggml_ctx, emb);
        emb      = time_embed_2->forward(ctx, emb);  // [N, time_embed_dim]

        // SDXL/SVD
        if (y != nullptr) {
            auto label_embed_0 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.0"]);
            auto label_embed_2 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.2"]);

            auto label_emb = label_embed_0->forward(ctx, y);
            label_emb      = ggml_silu_inplace(ctx->ggml_ctx, label_emb);
            label_emb      = label_embed_2->forward(ctx, label_emb);  // [N, time_embed_dim]

            emb = ggml_add(ctx->ggml_ctx, emb, label_emb);  // [N, time_embed_dim]
        }
        // sd::ggml_graph_cut::mark_graph_cut(emb, "unet.prelude", "emb");

        // input_blocks
        std::vector<ggml_tensor*> hs;

        // input block 0
        auto h = input_blocks_0_0->forward(ctx, x);
        sd::ggml_graph_cut::mark_graph_cut(h, "unet.input_blocks.0", "h");

        ggml_set_name(h, "bench-start");
        hs.push_back(h);
        // input block 1-11
        size_t len_mults    = channel_mult.size();
        int input_block_idx = 0;
        int ds              = 1;
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                h                = resblock_forward(name, ctx, h, emb, num_video_frames);  // [N, mult*model_channels, h, w]
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    h                = attention_layer_forward(name, ctx, h, context, num_video_frames);  // [N, mult*model_channels, h, w]
                }
                sd::ggml_graph_cut::mark_graph_cut(h, "unet.input_blocks." + std::to_string(input_block_idx), "h");
                hs.push_back(h);
            }
            if (tiny_unet) {
                input_block_idx++;
            }
            if (i != len_mults - 1) {
                ds *= 2;
                input_block_idx += 1;

                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                auto block       = std::dynamic_pointer_cast<DownSampleBlock>(blocks[name]);

                h = block->forward(ctx, h);  // [N, mult*model_channels, h/(2^(i+1)), w/(2^(i+1))]
                // sd::ggml_graph_cut::mark_graph_cut(h, "unet.input_blocks." + std::to_string(input_block_idx), "h");
                hs.push_back(h);
            }
        }
        // [N, 4*model_channels, h/8, w/8]

        // middle_block
        if (!tiny_unet) {
            h = resblock_forward("middle_block.0", ctx, h, emb, num_video_frames);  // [N, 4*model_channels, h/8, w/8]
            if (version != VERSION_SDXL_SSD1B && version != VERSION_SDXL_VEGA) {
                h = attention_layer_forward("middle_block.1", ctx, h, context, num_video_frames);  // [N, 4*model_channels, h/8, w/8]
                h = resblock_forward("middle_block.2", ctx, h, emb, num_video_frames);             // [N, 4*model_channels, h/8, w/8]
            }
        }
        sd::ggml_graph_cut::mark_graph_cut(h, "unet.middle_block", "h");
        if (controls.size() > 0) {
            auto cs = ggml_ext_scale(ctx->ggml_ctx, controls[controls.size() - 1], control_strength, true);
            h       = ggml_add(ctx->ggml_ctx, h, cs);  // middle control
        }
    return UnetHalfState{h, hs, emb, ds};
}

inline ggml_tensor* UnetModelBlock::forward_half1(
    GGMLRunnerContext* ctx,
    const UnetHalfState& st,
    ggml_tensor* context,
    int num_video_frames,
    std::vector<ggml_tensor*> controls,
    float control_strength) {
    auto out_0 = std::dynamic_pointer_cast<GroupNorm32>(blocks["out.0"]);
    auto out_2 = std::dynamic_pointer_cast<Conv2d>(blocks["out.2"]);

    ggml_tensor* h               = st.h;
    std::vector<ggml_tensor*> hs = st.hs;
    ggml_tensor* emb             = st.emb;
    int ds                       = st.ds;
    size_t len_mults             = channel_mult.size();

    int control_offset = static_cast<int>(controls.size() - 2);

        // output_blocks
        int output_block_idx = 0;
        for (int i = (int)len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                auto h_skip = hs.back();
                hs.pop_back();

                if (controls.size() > 0) {
                    auto cs = ggml_ext_scale(ctx->ggml_ctx, controls[control_offset], control_strength, true);
                    h_skip  = ggml_add(ctx->ggml_ctx, h_skip, cs);  // control net condition
                    control_offset--;
                }

                h = ggml_concat(ctx->ggml_ctx, h, h_skip, 2);

                std::string name = "output_blocks." + std::to_string(output_block_idx) + ".0";

                h = resblock_forward(name, ctx, h, emb, num_video_frames);

                int up_sample_idx = 1;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + ".1";

                    h = attention_layer_forward(name, ctx, h, context, num_video_frames);

                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) {
                    if (tiny_unet) {
                        output_block_idx++;
                        if (output_block_idx == 2) {
                            up_sample_idx = 1;
                        }
                    }
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + "." + std::to_string(up_sample_idx);
                    auto block       = std::dynamic_pointer_cast<UpSampleBlock>(blocks[name]);

                    h = block->forward(ctx, h);

                    ds /= 2;
                }

                output_block_idx += 1;
                sd::ggml_graph_cut::mark_graph_cut(h, "unet.output_blocks." + std::to_string(output_block_idx - 1), "h");
            }
        }

        // out
        h = out_0->forward(ctx, h);
        h = ggml_silu_inplace(ctx->ggml_ctx, h);
        h = out_2->forward(ctx, h);
        ggml_set_name(h, "bench-end");
        return h;  // [N, out_channels, h, w]
}

// ─── CF12-W7: N-way block-split forward over the linearized schedule ─────
//
// Purely additive: forward()/forward_half0/forward_half1 above are left
// byte-identical (they remain the proven monolithic + 2-way path, and the
// only path used by tiny-UNet variants). forward_range reproduces the same
// op sequence for an arbitrary contiguous block range [lo, hi); the CPU
// equivalence test (UNetModelRunner::test_split) checks that chaining
// forward_range segments reproduces forward() bit-for-bit before any GPU
// run. Callers must gate on num_split_blocks() > 0 (i.e. non-tiny).
inline void UnetModelBlock::forward_range(
    GGMLRunnerContext* ctx,
    UnetHalfState& io,
    int lo,
    int hi,
    ggml_tensor* x,
    ggml_tensor* timesteps,
    ggml_tensor* context,
    ggml_tensor* c_concat,
    ggml_tensor* y,
    int num_video_frames,
    std::vector<ggml_tensor*> controls,
    float control_strength) {
    const std::vector<BlockStep> sch = build_split_schedule();
    const int total = (int)sch.size();
    if (lo < 0)     lo = 0;
    if (hi > total) hi = total;

    ggml_tensor* h               = io.h;
    std::vector<ggml_tensor*> hs = io.hs;
    ggml_tensor* emb             = io.emb;

    int b = lo;
    if (lo == 0) {
        // ── prelude (mirrors forward_half0 head) ──────────────────────
        if (context != nullptr && context->ne[2] != x->ne[3]) {
            context = ggml_repeat(ctx->ggml_ctx, context,
                                  ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32,
                                                     context->ne[0], context->ne[1], x->ne[3]));
        }
        if (c_concat != nullptr) {
            if (c_concat->ne[3] != x->ne[3]) {
                c_concat = ggml_repeat(ctx->ggml_ctx, c_concat, x);
            }
            x = ggml_concat(ctx->ggml_ctx, x, c_concat, 2);
        }
        if (y != nullptr && y->ne[1] != x->ne[3]) {
            y = ggml_repeat(ctx->ggml_ctx, y,
                            ggml_new_tensor_2d(ctx->ggml_ctx, GGML_TYPE_F32, y->ne[0], x->ne[3]));
        }

        auto time_embed_0     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.0"]);
        auto time_embed_2     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.2"]);
        auto input_blocks_0_0 = std::dynamic_pointer_cast<Conv2d>(blocks["input_blocks.0.0"]);

        auto t_emb = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, model_channels);
        emb        = time_embed_0->forward(ctx, t_emb);
        emb        = ggml_silu_inplace(ctx->ggml_ctx, emb);
        emb        = time_embed_2->forward(ctx, emb);
        if (y != nullptr) {
            auto label_embed_0 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.0"]);
            auto label_embed_2 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.2"]);
            auto label_emb     = label_embed_0->forward(ctx, y);
            label_emb          = ggml_silu_inplace(ctx->ggml_ctx, label_emb);
            label_emb          = label_embed_2->forward(ctx, label_emb);
            emb                = ggml_add(ctx->ggml_ctx, emb, label_emb);
        }

        h = input_blocks_0_0->forward(ctx, x);
        sd::ggml_graph_cut::mark_graph_cut(h, "unet.input_blocks.0", "h");
        ggml_set_name(h, "bench-start");
        hs.clear();
        hs.push_back(h);
        b = 1;
    }

    for (; b < hi; ++b) {
        const BlockStep& s = sch[b];
        switch (s.kind) {
            case BlockStep::ConvIn:
                // only valid at b==0, handled by the prelude above.
                break;
            case BlockStep::InputRes: {
                std::string n0 = "input_blocks." + std::to_string(s.idx) + ".0";
                h              = resblock_forward(n0, ctx, h, emb, num_video_frames);
                if (s.has_attn) {
                    std::string n1 = "input_blocks." + std::to_string(s.idx) + ".1";
                    h              = attention_layer_forward(n1, ctx, h, context, num_video_frames);
                }
                sd::ggml_graph_cut::mark_graph_cut(h, "unet.input_blocks." + std::to_string(s.idx), "h");
                hs.push_back(h);
                break;
            }
            case BlockStep::InputDown: {
                std::string n0 = "input_blocks." + std::to_string(s.idx) + ".0";
                auto block     = std::dynamic_pointer_cast<DownSampleBlock>(blocks[n0]);
                h              = block->forward(ctx, h);
                hs.push_back(h);
                break;
            }
            case BlockStep::Middle: {
                h = resblock_forward("middle_block.0", ctx, h, emb, num_video_frames);
                if (s.has_attn) {
                    h = attention_layer_forward("middle_block.1", ctx, h, context, num_video_frames);
                    h = resblock_forward("middle_block.2", ctx, h, emb, num_video_frames);
                }
                sd::ggml_graph_cut::mark_graph_cut(h, "unet.middle_block", "h");
                if (!controls.empty()) {
                    auto cs = ggml_ext_scale(ctx->ggml_ctx, controls[controls.size() - 1], control_strength, true);
                    h       = ggml_add(ctx->ggml_ctx, h, cs);  // middle control
                }
                break;
            }
            case BlockStep::OutputRes: {
                auto h_skip = hs.back();
                hs.pop_back();
                if (!controls.empty()) {
                    // control_offset in forward_half1 starts at size-2 and
                    // decrements per output block → size-2-idx for block idx.
                    int control_offset = (int)controls.size() - 2 - s.idx;
                    if (control_offset >= 0 && control_offset < (int)controls.size()) {
                        auto cs = ggml_ext_scale(ctx->ggml_ctx, controls[control_offset], control_strength, true);
                        h_skip  = ggml_add(ctx->ggml_ctx, h_skip, cs);  // control net condition
                    }
                }
                h = ggml_concat(ctx->ggml_ctx, h, h_skip, 2);

                std::string n0 = "output_blocks." + std::to_string(s.idx) + ".0";
                h              = resblock_forward(n0, ctx, h, emb, num_video_frames);
                if (s.has_attn) {
                    std::string n1 = "output_blocks." + std::to_string(s.idx) + ".1";
                    h              = attention_layer_forward(n1, ctx, h, context, num_video_frames);
                }
                if (s.upsample) {
                    std::string nu = "output_blocks." + std::to_string(s.idx) + "." + std::to_string(s.up_sample_idx);
                    auto block     = std::dynamic_pointer_cast<UpSampleBlock>(blocks[nu]);
                    h              = block->forward(ctx, h);
                }
                sd::ggml_graph_cut::mark_graph_cut(h, "unet.output_blocks." + std::to_string(s.idx), "h");
                break;
            }
        }
    }

    if (hi == total) {
        auto out_0 = std::dynamic_pointer_cast<GroupNorm32>(blocks["out.0"]);
        auto out_2 = std::dynamic_pointer_cast<Conv2d>(blocks["out.2"]);
        h          = out_0->forward(ctx, h);
        h          = ggml_silu_inplace(ctx->ggml_ctx, h);
        h          = out_2->forward(ctx, h);
        ggml_set_name(h, "bench-end");
    }

    io.h   = h;
    io.hs  = std::move(hs);
    io.emb = emb;
    io.ds  = boundary_ds();  // informational; forward_range derives ds per block
}

struct UNetModelRunner : public GGMLRunner {
    UnetModelBlock unet;

    UNetModelRunner(ggml_backend_t backend,
                    ggml_backend_t params_backend,
                    const String2TensorStorage& tensor_storage_map,
                    const std::string prefix,
                    SDVersion version = VERSION_SD1)
        : GGMLRunner(backend, params_backend), unet(version, tensor_storage_map) {
        unet.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "unet";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
        unet.get_param_tensors(tensors, prefix);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                             const sd::Tensor<float>& timesteps_tensor,
                             const sd::Tensor<float>& context_tensor               = {},
                             const sd::Tensor<float>& c_concat_tensor              = {},
                             const sd::Tensor<float>& y_tensor                     = {},
                             int num_video_frames                                  = -1,
                             const std::vector<sd::Tensor<float>>& controls_tensor = {},
                             float control_strength                                = 0.f) {
        ggml_cgraph* gf = new_graph_custom(UNET_GRAPH_SIZE);

        ggml_tensor* x         = make_input(x_tensor);
        ggml_tensor* timesteps = make_input(timesteps_tensor);
        ggml_tensor* context   = make_optional_input(context_tensor);
        ggml_tensor* c_concat  = make_optional_input(c_concat_tensor);
        ggml_tensor* y         = make_optional_input(y_tensor);
        std::vector<ggml_tensor*> controls;
        controls.reserve(controls_tensor.size());
        for (const auto& control_tensor : controls_tensor) {
            controls.push_back(make_input(control_tensor));
        }

        if (num_video_frames == -1) {
            num_video_frames = static_cast<int>(x->ne[3]);
        }

        auto runner_ctx = get_context();

        ggml_tensor* out = unet.forward(&runner_ctx,
                                        x,
                                        timesteps,
                                        context,
                                        c_concat,
                                        y,
                                        num_video_frames,
                                        controls,
                                        control_strength);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    sd::Tensor<float> compute(int n_threads,
                              const sd::Tensor<float>& x,
                              const sd::Tensor<float>& timesteps,
                              const sd::Tensor<float>& context               = {},
                              const sd::Tensor<float>& c_concat              = {},
                              const sd::Tensor<float>& y                     = {},
                              int num_video_frames                           = -1,
                              const std::vector<sd::Tensor<float>>& controls = {},
                              float control_strength                         = 0.f) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // context: [N, max_position, hidden_size]([N, 77, 768]) or [1, max_position, hidden_size]
        // c_concat: [N, in_channels, h, w] or [1, in_channels, h, w]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(x, timesteps, context, c_concat, y, num_video_frames, controls, control_strength);
        };

        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
    }

    // ─── CF12-W6a: cross-rig UNet split runner methods ──────────────────────
    //
    // Carry-state extracted at the middle-block boundary. Stored host-side
    // (sd::Tensor<float>) so it can be serialized over the wire to a peer
    // rig that owns the output_blocks weights. The `ds` (downsample factor)
    // is reconstructible from `channel_mult.size()` and is not serialized.
    struct SplitCarry {
        sd::Tensor<float>              h;
        std::vector<sd::Tensor<float>> hs;
        sd::Tensor<float>              emb;
    };

    static constexpr const char* kSplitCacheH   = "unet.split.h";
    static constexpr const char* kSplitCacheEmb = "unet.split.emb";

    static std::string split_hs_key(size_t i) {
        return std::string("unet.split.hs.") + std::to_string(i);
    }

    ggml_cgraph* build_graph_half0(const sd::Tensor<float>& x_tensor,
                                   const sd::Tensor<float>& timesteps_tensor,
                                   const sd::Tensor<float>& context_tensor               = {},
                                   const sd::Tensor<float>& c_concat_tensor              = {},
                                   const sd::Tensor<float>& y_tensor                     = {},
                                   int num_video_frames                                  = -1,
                                   const std::vector<sd::Tensor<float>>& controls_tensor = {},
                                   float control_strength                                = 0.f) {
        ggml_cgraph* gf = new_graph_custom(UNET_GRAPH_SIZE);

        ggml_tensor* x         = make_input(x_tensor);
        ggml_tensor* timesteps = make_input(timesteps_tensor);
        ggml_tensor* context   = make_optional_input(context_tensor);
        ggml_tensor* c_concat  = make_optional_input(c_concat_tensor);
        ggml_tensor* y         = make_optional_input(y_tensor);
        std::vector<ggml_tensor*> controls;
        controls.reserve(controls_tensor.size());
        for (const auto& control_tensor : controls_tensor) {
            controls.push_back(make_input(control_tensor));
        }
        if (num_video_frames == -1) {
            num_video_frames = static_cast<int>(x->ne[3]);
        }

        auto runner_ctx = get_context();
        auto st         = unet.forward_half0(&runner_ctx, x, timesteps, context, c_concat, y,
                                             num_video_frames, controls, control_strength);

        runner_ctx.persist_cache_tensor(kSplitCacheH, st.h);
        runner_ctx.persist_cache_tensor(kSplitCacheEmb, st.emb);
        for (size_t i = 0; i < st.hs.size(); ++i) {
            runner_ctx.persist_cache_tensor(split_hs_key(i), st.hs[i]);
        }

        ggml_build_forward_expand(gf, st.h);
        return gf;
    }

    ggml_cgraph* build_graph_half1(const SplitCarry& carry,
                                   const sd::Tensor<float>& context_tensor               = {},
                                   int num_video_frames                                  = -1,
                                   const std::vector<sd::Tensor<float>>& controls_tensor = {},
                                   float control_strength                                = 0.f) {
        ggml_cgraph* gf = new_graph_custom(UNET_GRAPH_SIZE);

        ggml_tensor* h       = make_input(carry.h);
        ggml_tensor* emb     = make_input(carry.emb);
        ggml_tensor* context = make_optional_input(context_tensor);
        std::vector<ggml_tensor*> hs;
        hs.reserve(carry.hs.size());
        for (const auto& hs_tensor : carry.hs) {
            hs.push_back(make_input(hs_tensor));
        }
        std::vector<ggml_tensor*> controls;
        controls.reserve(controls_tensor.size());
        for (const auto& control_tensor : controls_tensor) {
            controls.push_back(make_input(control_tensor));
        }
        if (num_video_frames == -1) {
            num_video_frames = static_cast<int>(h->ne[3]);
        }

        UnetModelBlock::UnetHalfState st;
        st.h   = h;
        st.hs  = hs;
        st.emb = emb;
        st.ds  = unet.boundary_ds();

        auto runner_ctx  = get_context();
        ggml_tensor* out = unet.forward_half1(&runner_ctx, st, context,
                                              num_video_frames, controls, control_strength);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    bool read_cache_sd_tensor(const std::string& name, sd::Tensor<float>& out) {
        ggml_tensor* t = get_cache_tensor_by_name(name);
        if (t == nullptr) {
            return false;
        }
        out = sd::make_sd_tensor_from_ggml<float>(t);
        return true;
    }

    bool compute_half0(int n_threads,
                       const sd::Tensor<float>& x,
                       const sd::Tensor<float>& timesteps,
                       const sd::Tensor<float>& context,
                       const sd::Tensor<float>& c_concat,
                       const sd::Tensor<float>& y,
                       int num_video_frames,
                       const std::vector<sd::Tensor<float>>& controls,
                       float control_strength,
                       SplitCarry& carry_out) {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph_half0(x, timesteps, context, c_concat, y,
                                     num_video_frames, controls, control_strength);
        };
        auto result = GGMLRunner::compute<float>(get_graph, n_threads, false);
        if (!result.has_value()) {
            return false;
        }
        // result already corresponds to st.h via final_result; but we want the
        // host copy from cache_ctx for the explicit carry-state. Read all
        // persisted tensors back from cache.
        sd::Tensor<float> h_host;
        if (!read_cache_sd_tensor(kSplitCacheH, h_host)) {
            return false;
        }
        sd::Tensor<float> emb_host;
        if (!read_cache_sd_tensor(kSplitCacheEmb, emb_host)) {
            return false;
        }
        std::vector<sd::Tensor<float>> hs_host;
        for (size_t i = 0;; ++i) {
            sd::Tensor<float> hs_i;
            if (!read_cache_sd_tensor(split_hs_key(i), hs_i)) {
                break;
            }
            hs_host.push_back(std::move(hs_i));
        }
        carry_out.h   = std::move(h_host);
        carry_out.emb = std::move(emb_host);
        carry_out.hs  = std::move(hs_host);
        return true;
    }

    sd::Tensor<float> compute_half1(int n_threads,
                                    const SplitCarry& carry,
                                    const sd::Tensor<float>& context,
                                    int num_video_frames,
                                    const std::vector<sd::Tensor<float>>& controls,
                                    float control_strength,
                                    size_t output_dim_hint) {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph_half1(carry, context, num_video_frames, controls, control_strength);
        };
        return restore_trailing_singleton_dims(
            GGMLRunner::compute<float>(get_graph, n_threads, false),
            output_dim_hint);
    }

    // ─── CF12-W7: N-way range runner ─────────────────────────────────────
    //
    // Build the graph for blocks [lo, hi). lo==0 takes x/timesteps/c_concat/y
    // as inputs (carry_in ignored); lo>0 takes the carry tensors as inputs.
    // For the final stage (hi>=total) the graph expands on the noise_pred so
    // GGMLRunner::compute returns it directly. For intermediate stages we
    // persist h/emb/hs[] to the cache and expand on *every* carried tensor —
    // pass-through residuals from carry_in are graph leaves and would not be
    // reachable from h alone, so each must be an explicit graph root.
    ggml_cgraph* build_graph_range(const SplitCarry& carry_in,
                                   int lo,
                                   int hi,
                                   const sd::Tensor<float>& x_tensor,
                                   const sd::Tensor<float>& timesteps_tensor,
                                   const sd::Tensor<float>& context_tensor,
                                   const sd::Tensor<float>& c_concat_tensor,
                                   const sd::Tensor<float>& y_tensor,
                                   int num_video_frames,
                                   const std::vector<sd::Tensor<float>>& controls_tensor,
                                   float control_strength) {
        ggml_cgraph* gf = new_graph_custom(UNET_GRAPH_SIZE);
        const int total = unet.num_split_blocks();

        UnetModelBlock::UnetHalfState io;
        ggml_tensor* x         = nullptr;
        ggml_tensor* timesteps = nullptr;
        ggml_tensor* context   = make_optional_input(context_tensor);
        ggml_tensor* c_concat  = nullptr;
        ggml_tensor* y         = nullptr;

        if (lo == 0) {
            x         = make_input(x_tensor);
            timesteps = make_input(timesteps_tensor);
            c_concat  = make_optional_input(c_concat_tensor);
            y         = make_optional_input(y_tensor);
        } else {
            io.h   = make_input(carry_in.h);
            io.emb = make_input(carry_in.emb);
            io.hs.reserve(carry_in.hs.size());
            for (const auto& hs_tensor : carry_in.hs) {
                io.hs.push_back(make_input(hs_tensor));
            }
        }

        std::vector<ggml_tensor*> controls;
        controls.reserve(controls_tensor.size());
        for (const auto& control_tensor : controls_tensor) {
            controls.push_back(make_input(control_tensor));
        }

        if (num_video_frames == -1) {
            num_video_frames = (lo == 0) ? static_cast<int>(x->ne[3])
                                         : static_cast<int>(io.h->ne[3]);
        }

        auto runner_ctx = get_context();
        unet.forward_range(&runner_ctx, io, lo, hi, x, timesteps, context, c_concat, y,
                           num_video_frames, controls, control_strength);

        if (hi >= total) {
            ggml_build_forward_expand(gf, io.h);  // final: io.h == noise_pred
        } else {
            runner_ctx.persist_cache_tensor(kSplitCacheH, io.h);
            runner_ctx.persist_cache_tensor(kSplitCacheEmb, io.emb);
            for (size_t i = 0; i < io.hs.size(); ++i) {
                runner_ctx.persist_cache_tensor(split_hs_key(i), io.hs[i]);
            }
            ggml_build_forward_expand(gf, io.h);
            ggml_build_forward_expand(gf, io.emb);
            for (size_t i = 0; i < io.hs.size(); ++i) {
                ggml_build_forward_expand(gf, io.hs[i]);
            }
        }
        return gf;
    }

    // Run blocks [lo, hi). Final stage fills out_noise; intermediate stages
    // fill carry_out. Returns false on compute / cache-readback failure.
    bool compute_range(int n_threads,
                       int lo,
                       int hi,
                       const SplitCarry& carry_in,
                       const sd::Tensor<float>& x,
                       const sd::Tensor<float>& timesteps,
                       const sd::Tensor<float>& context,
                       const sd::Tensor<float>& c_concat,
                       const sd::Tensor<float>& y,
                       int num_video_frames,
                       const std::vector<sd::Tensor<float>>& controls,
                       float control_strength,
                       size_t output_dim_hint,
                       SplitCarry& carry_out,
                       sd::Tensor<float>& out_noise) {
        const int total = unet.num_split_blocks();
        auto get_graph  = [&]() -> ggml_cgraph* {
            return build_graph_range(carry_in, lo, hi, x, timesteps, context,
                                     c_concat, y, num_video_frames, controls, control_strength);
        };
        auto result = GGMLRunner::compute<float>(get_graph, n_threads, false);
        if (!result.has_value()) {
            return false;
        }
        if (hi >= total) {
            out_noise = restore_trailing_singleton_dims(std::move(result), output_dim_hint);
            return true;
        }
        if (!read_cache_sd_tensor(kSplitCacheH, carry_out.h)) {
            return false;
        }
        if (!read_cache_sd_tensor(kSplitCacheEmb, carry_out.emb)) {
            return false;
        }
        carry_out.hs.clear();
        for (size_t i = 0;; ++i) {
            sd::Tensor<float> hs_i;
            if (!read_cache_sd_tensor(split_hs_key(i), hs_i)) {
                break;
            }
            carry_out.hs.push_back(std::move(hs_i));
        }
        return true;
    }

    void test() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
        params.mem_buffer = nullptr;
        params.no_alloc   = false;

        ggml_context* ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);

        {
            // CPU, num_video_frames = 1, x{num_video_frames, 8, 8, 8}: Pass
            // CUDA, num_video_frames = 1, x{num_video_frames, 8, 8, 8}: Pass
            // CPU, num_video_frames = 3, x{num_video_frames, 8, 8, 8}: Wrong result
            // CUDA, num_video_frames = 3, x{num_video_frames, 8, 8, 8}: nan
            int num_video_frames = 3;

            sd::Tensor<float> x({8, 8, 8, num_video_frames});
            std::vector<float> timesteps_vec(num_video_frames, 999.f);
            auto timesteps = sd::Tensor<float>::from_vector(timesteps_vec);
            x.fill_(0.5f);
            // print_ggml_tensor(x);

            sd::Tensor<float> context({1024, 1, num_video_frames});
            context.fill_(0.5f);
            // print_ggml_tensor(context);

            sd::Tensor<float> y({768, num_video_frames});
            y.fill_(0.5f);
            // print_ggml_tensor(y);

            sd::Tensor<float> out;

            int64_t t0   = ggml_time_ms();
            auto out_opt = compute(8,
                                   x,
                                   timesteps,
                                   context,
                                   {},
                                   y,
                                   num_video_frames,
                                   {},
                                   0.f);
            int64_t t1   = ggml_time_ms();

            GGML_ASSERT(!out_opt.empty());
            out = std::move(out_opt);
            print_sd_tensor(out);
            LOG_DEBUG("unet test done in %lldms", t1 - t0);
        }
    }
};

#endif  // __UNET_HPP__
