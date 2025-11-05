# MLX Swift Examples

Example [MLX Swift](https://github.com/ml-explore/mlx-swift) programs.  The language model
examples use models implemented in [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).

- [MNISTTrainer](Applications/MNISTTrainer/README.md): An example that runs on
  both iOS and macOS that downloads MNIST training data and trains a
  [LeNet](https://en.wikipedia.org/wiki/LeNet).

- [LLMEval](Applications/LLMEval/README.md): An example that runs on both iOS
  and macOS that downloads an LLM and tokenizer from Hugging Face and
  generates text from a given prompt.

- [VLMEval](Applications/VLMEval/README.md): An example that runs on iOS, macOS and visionOS to download a VLM and tokenizer from Hugging Face and
  analyzes the given image and describe it in text.

- [MLXChatExample](Applications/MLXChatExample/README.md): An example chat app that runs on both iOS and macOS that supports LLMs and VLMs.

- [LoRATrainingExample](Applications/LoRATrainingExample/README.md): An example that runs on macOS that downloads an LLM and fine-tunes it using LoRA (Low-Rank Adaptation) with training data.

- [LinearModelTraining](Tools/LinearModelTraining/README.md): An example that
  trains a simple linear model.

- [StableDiffusionExample](Applications/StableDiffusionExample/README.md): An
  example that runs on both iOS and macOS that downloads a stable diffusion model
  from Hugging Face and  and generates an image from a given prompt.

- [llm-tool](Tools/llm-tool/README.md): A command line tool for generating text
  using a variety of LLMs available on the Hugging Face hub.

- [ExampleLLM](Tools/ExampleLLM/README.md): A command line tool using the simplified API to interact with LLMs.

- [image-tool](Tools/image-tool/README.md): A command line tool for generating images
  using a stable diffusion model from Hugging Face.

- [mnist-tool](Tools/mnist-tool/README.md): A command line tool for training a
  a LeNet on MNIST.
  
> [!IMPORTANT]
> `MLXLMCommon`, `MLXLLM`, `MLXVLM` and `MLXEmbedders` have moved to a new repository
> containing _only_ reusable libraries: [mlx-swift-lm](https://github.com/ml-explore/mlx-swift).

Previous URLs and tags will continue to work, but going forward all updates to these
libraries will be done in the other repository.  Previous tags _are_ supported in
the new repository.

> [!TIP]
> Contributors that wish to edit both `mlx-swift-examples` and `mlx-swift-lm` can
> use [this technique in Xcode](https://developer.apple.com/documentation/xcode/editing-a-package-dependency-as-a-local-package).


# Reusable Libraries

LLM and VLM implementations are available in [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm):

- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon) -- common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm) -- large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm) -- vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders) -- popular Encoders / Embedding models example implementations

`mlx-swift-examples` also contains a few reusable libraries that can be imported with this code in your `Package.swift` or by referencing the URL in Xcode:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-examples/", branch: "main"),
```

Then add one or more libraries to the target as a dependency:

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "StableDiffusion", package: "mlx-libraries")
    ]),
```

- [StableDiffusion](https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/stablediffusion) -- SDXL Turbo and Stable Diffusion model example implementations
- [MLXMNIST](https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/mlxmnist) -- MNIST implementation for all your digit recognition needs

## Running

The application and command line tool examples can be run from Xcode or from
the command line:

```
./mlx-run llm-tool --prompt "swift programming language"
```

Note: `mlx-run` is a shell script that uses `xcode` command line tools to
locate the built binaries. It is equivalent to running from Xcode itself.

See also:

- [MLX troubleshooting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/troubleshooting)
