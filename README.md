# MLX Swift Examples

Example [MLX Swift](https://github.com/ml-explore/mlx-swift) programs.

- [MNISTTrainer](Applications/MNISTTrainer/README.md): An example that runs on
  both iOS and macOS that downloads MNIST training data and trains a
  [LeNet](https://en.wikipedia.org/wiki/LeNet). 

- [LLMEval](Applications/LLMEval/README.md): An example that runs on both iOS
  and macOS that downloads an LLM and tokenizer from Hugging Face and  and
  generates text from a given prompt. 

- [LinearModelTraining](Tools/LinearModelTraining/README.md): An example that
  trains a simple linear model.

- [llm-tool](Tools/llm-tool/README.md): A command line tool for generating text
  using a variety of LLMs available on the Hugging Face hub.

- [mnist-tool](Tools/mnist-tool/README.md): A command line tool for training a
  a LeNet on MNIST.


## Installation of MLXLLM and MLXMNIST libraries

The MLXLLM and MLXMNIST libraries in the example repo are available as Swift Packages.


Add the following dependency to your Package.swift

```swift  
.package(url: "https://github.com/ml-explore/mlx-swift-examples/", branch: "main"),
```

Then add one library or both libraries to the target as a dependency. 

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "LLM", package: "mlx-swift-examples")
    ]),
```

Alternatively, add `https://github.com/ml-explore/mlx-swift-examples/` to the `Project Dependencies` and set the `Dependency Rule` to `Branch` and `main` in Xcode. 
