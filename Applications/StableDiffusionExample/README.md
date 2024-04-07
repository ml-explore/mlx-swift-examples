#  StableDiffusionExample

An example application that runs the StableDiffusion example code.

See also [image-tool](../../Tools/image-tool) for a command line example.

This example application accepts a prompt and used the StableDiffusion example
library to render an image using:

- [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo)

Please refer to that model for license and other information.

If you are interested in adjusting the generated images, look in 
[ContentView.swift](ContentView.swift) at this method:

```swift
    func generate(prompt: String, negativePrompt: String, showProgress: Bool) async 
```

### Troubleshooting

Stable diffusion can run in less that 4G available memory (typically a
device or computer with 6G of memory or more) in a constrained mode -- it will
load and unload parts of the model as it runs and it can only perform one step
of diffusion.  This is configured automatically, see `modelFactory.conserveMemory`
in [ContentView.swift](ContentView.swift).

On a device or computer with more memory the model will be kept resident and
images can be regenerated much more efficiently.

If the program exits while generating the image it may have exceeded the available
memory.
