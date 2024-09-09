#  Stable Diffusion

Stable Diffusion in MLX. The implementation was ported from Hugging Face's
[diffusers](https://huggingface.co/docs/diffusers/index) and 
[mlx-examples/stable_diffusion](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion).
Model weights are downloaded directly from the Hugging Face hub. The implementation currently
supports the following models:

- [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo)
- [stabilitiai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)

## Usage

See [StableDiffusionExample](../../Applications/StableDiffusionExample) and
[image-tool](../../Tools/image-tool) for examples of using this code.

The basic sequence is:

- download & load the model
- generate latents
- evaluate the latents one by one
- decode the last latent generated
- you have an image!

```swift
let configuration = StableDiffusionConfiguration.presetSDXLTurbo

let generator = try configuration.textToImageGenerator(
    configuration: model.loadConfiguration)

generator.ensureLoaded()

// generate the latents -- these are the iterations for generating
// the output image.  this is just generating the evaluation graph
let parameters = generate.evaluateParameters(configuration: configuration)
let latents = generator.generateLatents(parameters: parameters)

// evaluate the latents (evalue the graph) and keep the last value generated
var lastXt: MLXArray?
for xt in latents {
    eval(xt)
    lastXt = xt
}

// decode the final latent into an image
if let lastXt {
    var raster = decoder(lastXt[0])
    raster = (image * 255).asType(.uint8).squeezed()
    eval(raster)
    
    // turn it into a CGImage
    let image = Image(raster).asCGImage()
    
    // or write it out
    try Image(raster).save(url: url)
}
```
