# Distributed per-layer UNet execution

The C API in [`include/stable-diffusion.h`](../include/stable-diffusion.h) exposes
a single UNet forward pass as a chain of contiguous *block ranges*. A scheduler can
run those ranges back to back in one process, or spread them across several rigs, and
can drive the whole denoise loop through a remote-UNet callback.

This is a strict, behavior-preserving extension of the monolithic UNet path: the
split code is the same `forward()` walked in pieces, sharing the identical math,
carry-state `{h, hs, emb}` handling and fp16 boundaries. A single whole-range call is
bit-for-bit equivalent to the un-split forward; tiling that range into N stages
reassembles the exact same result.

Applies to UNet backbones (SD1.x / SD2 / SDXL). DiT/MMDiT/Flux and tiny-UNet variants
report a block count of 0 and must use the whole-UNet path.

## Block model

`forward()` is a linear sequence of blocks:

| Index range            | Blocks                | Skip stack `hs[]`          |
| ---------------------- | --------------------- | -------------------------- |
| `[0]`                  | `conv_in`             | pushes `hs[0]`             |
| `[1 .. n_in-1]`        | input res / down      | each pushes a skip         |
| `[n_in]`               | middle (non-tiny)     | —                          |
| `[n_in+1 .. count-1]`  | output res / up       | each pops a skip           |

The final out-conv is folded onto the last stage. Every per-block detail (downsample
factor, layer names, attention placement, controlnet offsets) is a pure function of
the block index, so a range is fully reconstructible from the carry-state `{h, hs, emb}`
with no extra wire fields.

`sd_unet_block_count()` returns the total number of splittable blocks (SD1.x/SD2 → 25,
SDXL → 19), or 0 for non-split-capable backbones. `sd_loaded_backbone_tag()` returns
the backbone family string for capability advertisement.

## Block-range contract

Ranges are half-open: `[block_lo, block_hi)` runs blocks `block_lo .. block_hi-1`.

- `block_lo == 0` seeds the prelude (time/label embedding + `conv_in`) from the staged
  `x` / `timesteps` / `context` / `c_concat` / `y`; no carry-in is read.
- `block_lo > 0` reads the carry `{h, hs, emb}` staged on the state.
- `block_hi >= block_count` appends the final out-conv and produces the noise-pred,
  read via `sd_split_state_get_output`.
- Otherwise the range produces a carry `{h, hs, emb}`, read via
  `sd_split_state_get_carry_tensor`, for the next stage.

Down-path skip residuals ride the `hs[]` stack forward across intermediate stages until
the up-path pops them, so **any** tiling of `[0, block_count)` is valid.

## Tensor layout

All tensors crossing this surface are `sd::Tensor` shapes in **WHCN** order (ggml
convention: dim 0 = width, 1 = height, 2 = channels, 3 = batch) — **not** NCHW. Shapes
are passed as `int64` arrays whose element `[0]` is the fastest-varying (width) axis.
Data is plain contiguous fp32.

## Driving a single block range

Each denoise step reuses one `sd_split_state_t`. The first range (`block_lo == 0`)
consumes the staged inputs; later ranges consume the carry produced by the previous one.

```c
sd_split_state_t* st = sd_split_state_new();

// Stage the per-step UNet inputs (WHCN shapes).
sd_split_state_set_input(st, "x",         x_data,   x_shape,   x_ndim);
sd_split_state_set_input(st, "timesteps", t_data,   t_shape,   t_ndim);
sd_split_state_set_input(st, "context",   ctx_data, ctx_shape, ctx_ndim);
// c_concat / y are optional: pass NULL / NULL / 0 if the backbone doesn't use them.

const int count = sd_unet_block_count(ctx);
const int cut   = count / 2;            // any 0 < cut < count works

// Stage 1: blocks [0, cut) -> leaves a carry on the state.
sd_compute_unet_split_range(ctx, 0, cut, step_idx, total_steps, st);

// (Optionally read back {h, hs, emb} here and ship them to another rig; the
//  receiver stages them with sd_split_state_set_hs_count + set_carry_tensor.)

// Stage 2: blocks [cut, count) -> fills the noise-pred.
sd_compute_unet_split_range(ctx, cut, count, step_idx, total_steps, st);

const float*   eps;
const int64_t* eps_shape;
int            eps_ndim;
sd_split_state_get_output(st, &eps, &eps_shape, &eps_ndim);
// `eps` is the noise prediction for this step (WHCN, borrowed).

sd_split_state_free(st);
```

`sd_compute_unet_split_step()` is a thin 2-way convenience over the same path:
`which_half = 0` runs `[0, mid_cut)` and `which_half = 1` runs `[mid_cut, count)`,
with the cut at the middle-block boundary.

### Moving the carry between rigs

To hop a carry across a process boundary, read the tensors on the producing rig and
re-stage them on the consuming rig:

```c
// Producer (after a non-final range):
int hs_count;
sd_split_state_get_carry_count(st, &hs_count);
// read "h", "emb", "hs.0" .. "hs.<hs_count-1>" via sd_split_state_get_carry_tensor,
// or use sd_split_state_serialize(st, &blob, &nbytes) for one self-describing buffer.

// Consumer (before a block_lo>0 range):
sd_split_state_set_hs_count(st2, hs_count);
// stage "h", "emb", "hs.*" via sd_split_state_set_carry_tensor,
// or sd_split_state_deserialize(blob, nbytes, &st2).
```

## Remote-denoise callback

`sd_set_remote_unet_cb()` installs a per-step UNet delegate. The host then runs the
full `sample()` loop locally — text-encode, denoiser scalings, sigmas, sampler and VAE
are all unchanged — but routes each per-step UNet eval through the callback, which can
drive the N-way block chain across rigs and return the eps.

```c
static int my_remote_unet(void* user,
                          const float* x,   const int64_t* x_ne,   int x_ndim,
                          float timestep,
                          const float* ctx, const int64_t* ctx_ne, int ctx_ndim,
                          const float* y,   const int64_t* y_ne,   int y_ndim,
                          float** out_eps,  int64_t* out_ne,       int* out_ndim) {
    // Coordinate the block-range chain across rigs for this (x, timestep, ctx, y),
    // then return the eps:
    //   *out_eps = malloc(numel * sizeof(float));   // host frees this
    //   fill out_ne[0 .. *out_ndim)                 // WHCN; out_ne has >= 8 slots
    //   *out_ndim = ...
    return 0;  // non-zero aborts the step
}

sd_set_remote_unet_cb(ctx, my_remote_unet, /*user=*/NULL);
// ... call the normal txt2img/img2img generation ...
sd_set_remote_unet_cb(ctx, NULL, NULL);   // clear when done
```

The callback is invoked once per conditioning eval — cond and uncond separately under
CFG. `x` is the scaled UNet input, `ctx` the `c_crossattn` prompt embeds, and `y` the
`c_vector` (SDXL pooled; `ndim == 0` when absent). All shapes are WHCN. The callee
allocates `*out_eps` with `malloc` (the host frees it) and fills `out_ne[0..*out_ndim)`.

## Error codes

`sd_compute_unet_split_range` / `sd_compute_unet_split_step` and the serialise helpers
return one of:

| Code               | Meaning                                                       |
| ------------------ | ------------------------------------------------------------- |
| `SD_SPLIT_OK`      | success                                                       |
| `SD_SPLIT_EINVAL`  | null/invalid argument or out-of-range block index            |
| `SD_SPLIT_ENOTSUP` | backbone is not split-capable (DiT/Flux/tiny-UNet)            |
| `SD_SPLIT_ESTATE`  | carry-state mismatch (e.g. `block_lo > 0` without a carry)    |
| `SD_SPLIT_EALLOC`  | allocation failure                                            |
