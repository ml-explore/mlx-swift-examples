# image-tool

Command line tool for generating images with [Stable Diffusion](https://github.com/ml-explore/mlx-swift-examples/tree/main/Libraries/StableDiffusion) on MLX.

Two subcommands:

- `sd text` — text-to-image
- `sd image` — image-to-image (init from an existing image)

### Building

Build the `image-tool` scheme in Xcode.

### Running: Xcode

To run this in Xcode press cmd-opt-r to set the scheme arguments. For example:

```
sd text --prompt "purple cow on the moon" --output /tmp/cow.png
```

Then cmd-r to run.

> Note: you may be prompted for access to your Documents directory — this is where
> the Hugging Face HubApi stores downloaded model files.

### Running: Command Line

Use the `mlx-run` script to run the command line tool:

```
./mlx-run image-tool sd text --prompt "purple cow on the moon" --output /tmp/cow.png
```

By default this will find and run the tool built in _Release_ configuration. Specify `--debug`
to find and run the tool built in _Debug_ configuration.

### Models

Pass `--model` to pick a preset (default: `sdxlTurbo`). The available presets are defined by
`StableDiffusionConfiguration.Preset` in
[Libraries/StableDiffusion](https://github.com/ml-explore/mlx-swift-examples/tree/main/Libraries/StableDiffusion)
and include SD 2.1 Base and SDXL Turbo variants.

### Text to image

Generate an image from a prompt:

```
./mlx-run image-tool sd text \
    --prompt "an astronaut riding a horse on mars, cinematic" \
    --negative-prompt "low quality, blurry" \
    --steps 4 \
    --output /tmp/out.png
```

Useful options:

| Option | Description |
| --- | --- |
| `--prompt` | Text prompt (default: `"purple cow on the moon"`) |
| `--negative-prompt` | Negative prompt (requires `--cfg` > 1) |
| `--cfg` | Classifier-free-guidance weight |
| `--steps` | Number of denoising steps |
| `--image-count` | Number of images to generate |
| `--batch-size` | Decoding batch size |
| `--latent-width` / `--latent-height` | Latent size (output is 8× these values) |
| `--rows` | Number of rows when laying out multiple images into a grid |
| `--seed` | PRNG seed for reproducible output |
| `--output` | Output PNG path (default: `/tmp/out.png`) |

### Image to image

Start from an existing image:

```
./mlx-run image-tool sd image \
    --input /tmp/in.png \
    --prompt "...same image but in watercolor..." \
    --strength 0.7 \
    --output /tmp/out.png
```

Additional options:

| Option | Description |
| --- | --- |
| `--input` | Input image path (required) |
| `--max-edge` | Scale input so its longest edge is this many pixels (default: 1024) |
| `--strength` | Noise strength — higher means more deviation from the input (default: 0.9) |

### Memory

The `--memory-stats`, `--cache-size`, and `--memory-size` options control MLX's memory limits and
print before/after snapshots; see
[`MemoryArguments`](https://github.com/ml-explore/mlx-swift-examples/blob/main/Tools/image-tool/Arguments.swift)
for the same options shared with `llm-tool`.

### Float16 and quantization

By default the model loads in float16. Use `--no-float16` to disable float16 conversion, or
`--quantize` to enable quantization. See
[`LoadConfiguration`](https://github.com/ml-explore/mlx-swift-examples/tree/main/Libraries/StableDiffusion)
for details.

### See also

- [llm-tool](../llm-tool/README.md) — equivalent command line tool for language models
- [StableDiffusion](https://github.com/ml-explore/mlx-swift-examples/tree/main/Libraries/StableDiffusion) — underlying library
- [StableDiffusionExample](../../Applications/StableDiffusionExample/) — SwiftUI example app built on the same library
