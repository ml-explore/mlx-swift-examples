#  VLMEval

An example that:

- downloads a vision language model (Qwen-VL-2B)
- processes an image with a prompt
- displays the analyzed results in JSON format

> Note: this _must_ be built Release, otherwise you will encounter
stack overflows.

You will need to set the Team on the VLMEval target in order to build and
run on macOS.

Some notes about the setup:

- This downloads models from hugging face so VLMEval -> Signing & Capabilities has the "Outgoing Connections (Client)" set in the App Sandbox
- VLM models are large so this uses significant memory
- The Qwen-VL-2B 4-bit model is optimized for performance while maintaining quality
- The example processes images and provides detailed analysis including:
  - Description of the image
  - List of main objects
  - Dominant colors
  - Lighting conditions
  - Compositional analysis

### Image Processing

The example application uses Qwen-VL-2B model by default, see [ContentView.swift](ContentView.swift):

```swift
self.modelContainer = try await VLMModelFactory.shared.loadContainer(
    configuration: ModelRegistry.qwen2VL2BInstruct4Bit)
```

The application:
1. Downloads a sample image
2. Processes it through the vision language model
3. Describes the images based on the prompt, providing detailed analysis of the content, objects, colors, and composition.

### Troubleshooting

If the program crashes with a very deep stack trace you may need to build
in Release configuration. This seems to depend on the size of the model.

There are a couple options:

- Build Release
- Force the model evaluation to run on the main thread, e.g. using @MainActor
- Build `Cmlx` with optimizations by modifying `mlx/Package.swift` and adding `.unsafeOptions(["-O3"]),`

Building in Release / optimizations will remove a lot of tail calls in the C++ 
layer. These lead to the stack overflows.

### Performance

The application is optimized for:
- Efficient image processing
- Token generation with a limit of 800 tokens
- Real-time JSON parsing and display

You may find that running outside the debugger boosts performance. You can do this in Xcode by pressing cmd-opt-r and unchecking "Debug Executable".
