# Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/mlxlmcommon/porting)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/mlxlmcommon) -- common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/mlxllm) -- large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/mlxvlm) -- vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/mlxembedders) -- popular Encoders / Embedding models example implementations
- [StableDiffusion](https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/stablediffusion) -- SDXL Turbo and Stable Diffusion model example implementations
- [MLXMNIST](https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/mlxmnist) -- MNIST implementation for all your digit recognition needs

# MLX Swift Examples

Example [MLX Swift](https://github.com/ml-explore/mlx-swift) programs.

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

## Interacting with LLMs

See also [MLXLMCommon](Libraries/MLXLMCommon).  You can easily use
a wide variety of open weight LLM and VLMs in your code.  You can use
this simplified API:

```swift
let model = try await loadModel(id: "mlx-community/Qwen3-4B-4bit")
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?")
print(try await session.respond(to: "How about a great place to eat?")
```

Or use the underlying API to control every aspect of the evaluation.

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

## Installation of libraries

The MLXLLM, MLXVLM, MLXLMCommon, MLXMNIST, MLXEmbedders, and StableDiffusion libraries in the example repo are available
as Swift Packages.


Add the following dependency to your Package.swift

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-examples/", branch: "main"),
```

Then add one or more libraries to the target as a dependency:

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-examples")
    ]),
```

Alternatively, add `https://github.com/ml-explore/mlx-swift-examples/` to the `Project Dependencies` and set the `Dependency Rule` to `Branch` and `main` in Xcode.
