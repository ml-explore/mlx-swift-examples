# ``MLXLLM``

Example implementations of various Large Language Models (LLMs).

## Other MLX Libraries Packages

- [MLXEmbedders](MLXEmbedders)
- [MLXLLM](MLXLLM)
- [MLXLMCommon](MLXLMCommon)
- [MLXMNIST](MLXMNIST)
- [MLXVLM](MLXVLM)
- [StableDiffusion](StableDiffusion)

## Quick Start

See <doc:evaluation>.

Using LLMs and VLMs is as easy as this:

```swift
let model = try await loadModel(id: "mlx-community/Qwen3-4B-4bit")
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?")
print(try await session.respond(to: "How about a great place to eat?")
```

More advanced APIs are available for those that need them, see <doc:using-model>.

## Topics

- <doc:evaluation>
- <doc:adding-model>
- <doc:using-model>

### Models

- ``CohereModel``
- ``GemmaModel``
- ``Gemma2Model``
- ``InternLM2Model``
- ``LlamaModel``
- ``OpenELMModel``
- ``PhiModel``
- ``Phi3Model``
- ``PhiMoEModel``
- ``Qwen2Model``
- ``Qwen3Model``
- ``Starcoder2Model``
- ``MiMoModel``
- ``GLM4Model``
- ``AceReason``
