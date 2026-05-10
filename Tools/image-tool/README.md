# image-tool

A command line tool for generating images with the Stable Diffusion example
library.

See additional documentation:

- [StableDiffusion](../../Libraries/StableDiffusion/README.md) -- reusable
  Stable Diffusion implementation
- [StableDiffusionExample](../../Applications/StableDiffusionExample/README.md)
  -- SwiftUI application using the same library

### Building

Build the `image-tool` scheme in Xcode.

### Running: Xcode

Configure the scheme arguments (Product > Scheme > Edit Scheme > Run >
Arguments). For example:

```
sd text \
    --prompt "purple cow on the moon" \
    --output /tmp/out.png
```

Then press <kbd>⌘</kbd>+<kbd>R</kbd> to run.

> Note: the first run downloads model weights from Hugging Face. You may be
prompted for access to the download location used by the Hugging Face `HubApi`.

### Running: Command Line

Use the `mlx-run` script to run the command line tools:

```
./mlx-run image-tool sd text --prompt "purple cow on the moon" --output /tmp/out.png
```

By default this will find and run the tool built in _Release_ configuration.
Specify `--debug` to find and run the tool built in _Debug_ configuration:

```
./mlx-run --debug image-tool sd text --prompt "a watercolor robot" --output /tmp/out.png
```

### Text to Image

The `sd text` command generates one or more images from a prompt and writes an
image file. The output format is inferred from the `--output` extension and
defaults to PNG when the extension is not recognized:

```
./mlx-run image-tool sd text \
    --model sdxl-turbo \
    --prompt "a small cabin beside a frozen lake" \
    --steps 4 \
    --seed 1234 \
    --output /tmp/cabin.png
```

Common generation options include:

- `--prompt` and `--negative-prompt` for text conditioning
- `--image-count`, `--batch-size`, and `--rows` for image grids
- `--latent-width` and `--latent-height` for output size, where the final image
  dimensions are 8x the latent dimensions
- `--cfg`, `--steps`, and `--seed` for generation behavior

### Image to Image

The `sd image` command uses an input image as the starting point:

```
./mlx-run image-tool sd image \
    --input support/test.jpg \
    --prompt "a studio portrait in watercolor" \
    --strength 0.75 \
    --output /tmp/portrait.png
```

The input image is loaded at dimensions compatible with the model before
generation. Higher `--strength` values allow the prompt to change more of the
original image.

### Models and Memory

The default model is `sdxl-turbo`. Use `--model base` for the Stable Diffusion
2.1 base preset. Pass `--no-float16` to disable float16 conversion or
`--quantize` to enable quantization.

Memory can be constrained with `--cache-size` and `--memory-size`, both in
megabytes.

See also:

- [MLX troubleshooting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/troubleshooting)
