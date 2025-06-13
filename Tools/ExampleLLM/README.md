#  ExampleLLM

An example that uses the simplified APIs to load and evaluate an LLM in only a few lines of
code:

```swift
let model = try await loadModel(id: "mlx-community/Qwen3-4B-4bit")
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?")
print(try await session.respond(to: "How about a great place to eat?")
```

See various READMEs:

- [MLXLMCommon](../../Libraries/MLXLMCommon/README.md) -- common LM code
- [MLXLLM](../../Libraries/MLXLLM/README.md) -- large language models
- [MLXVLM](../../Libraries/MLXVLM/README.md) -- vision language models

### Building

Build the `ExampleLLM` scheme in Xcode.

### Running: Xcode

Just press cmd-r to run!

### Running: Command Line

Use the `mlx-run` script to run the command line tools:

```
./mlx-run ExampleLLM
```

Note: `mlx-run` is a shell script that uses `xcode` command line tools to
locate the built binaries.  It is equivalent to running from Xcode itself.

By default this will find and run the tools built in _Release_ configuration.  Specify `--debug`
to find and run the tool built in _Debug_ configuration.

