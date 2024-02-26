// Copyright © 2024 Apple Inc.

import AsyncAlgorithms
import Foundation
import Hub
import MLX
import MLXNN
import MLXRandom
import Tokenizers

struct LLMError: Error {
    let message: String
}

/// Load and return the model and tokenizer
public func load(
    hub: HubApi = HubApi(), name: String, progressHandler: @escaping (Progress) -> Void = { _ in }
) async throws -> (LLMModel, Tokenizer) {
    // note: this doesn't have a way to pass the HubApi
    let tokenizer = try await loadTokenizer(name: name)

    // download the model weights and config
    let repo = Hub.Repo(id: name)
    let modelFiles = ["config.json", "weights.00.safetensors"]
    let modelDirectory = try await hub.snapshot(
        from: repo, matching: modelFiles, progressHandler: progressHandler)

    // create the model (no weights loaded)
    let configurationURL = modelDirectory.appending(component: "config.json")
    let baseConfig = try JSONDecoder().decode(
        BaseConfiguration.self, from: Data(contentsOf: configurationURL))

    let model = try baseConfig.modelType.createModel(configuration: configurationURL)

    // load the weights
    let weights = try loadArrays(url: modelDirectory.appending(component: "weights.00.safetensors"))

    // quantize if needed
    if let quantization = baseConfig.quantization {
        quantizeIfNeeded(model: model, weights: weights, quantization: quantization)
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)

    return (model, tokenizer)
}

public func loadTokenizer(name: String) async throws -> Tokenizer {
    // from AutoTokenizer.from() -- this lets us override parts of the configuration
    let config = LanguageModelConfigurationFromHub(modelName: name)
    guard var tokenizerConfig = try await config.tokenizerConfig else {
        throw LLMError(message: "missing config")
    }
    let tokenizerData = try await config.tokenizerData

    if let tokenizerClass = tokenizerConfig.tokenizerClass?.stringValue,
        let replacement = replacementTokenizers[tokenizerClass]
    {
        var dictionary = tokenizerConfig.dictionary
        dictionary["tokenizer_class"] = replacement
        tokenizerConfig = Config(dictionary)
    }

    return try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
}

/// overrides for TokenizerModel/knownTokenizers
let replacementTokenizers = [
    "CodeLlamaTokenizer": "LlamaTokenizer"
]

private func quantizeIfNeeded(
    model: LLMModel, weights: [String: MLXArray], quantization: BaseConfiguration.Quantization
) {

    func linearPredicate(layer: Module) -> Bool {
        if let layer = layer as? Linear {
            // avoid quantizing gate layers, otherwise we have to re-quant and upload all the mixtral models
            return layer.weight.dim(0) != 8
        }
        return false
    }

    var predicate = linearPredicate(layer:)

    // for legacy models that don't have lm_head quant due to non-32 dims
    if weights["lm_head.scales"] == nil {
        let vocabularySize = model.vocabularySize

        func vocabularySizePredicate(layer: Module) -> Bool {
            if let layer = layer as? Linear {
                return layer.weight.dim(0) != 8 && layer.weight.dim(0) != vocabularySize
            }
            return false
        }

        predicate = vocabularySizePredicate(layer:)
    }

    QuantizedLinear.quantize(
        model: model, groupSize: quantization.groupSize, bits: quantization.bits,
        predicate: predicate)
}

private func sample(logits: MLXArray, temp: Float) -> MLXArray {
    if temp == 0 {
        return argMax(logits, axis: -1)
    } else {
        return categorical(logits * (1 / temp))
    }
}

/// Synchronous generator of tokens.
///
/// Port of `generate_step()` from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py
public struct TokenIterator: Sequence, IteratorProtocol {
    let model: LLMModel
    let temp: Float

    var y: MLXArray
    var cache: [(MLXArray, MLXArray)]

    var first = true

    public init(prompt: MLXArray, model: LLMModel, temp: Float = 0.0) {
        self.model = model
        self.temp = temp
        self.y = prompt
        self.cache = []
    }

    mutating public func next() -> MLXArray? {
        var logits: MLXArray
        (logits, cache) = model(expandedDimensions(y, axis: 0), cache: cache.isEmpty ? nil : cache)
        y = sample(logits: logits[-1, axis: 1], temp: temp)

        return y
    }
}

/// Async generator of tokens.
///
/// Port of `generate_step()` from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py.
///
/// Note that because MLXArray is not thread safe this eval's the result and sends the TokenId back
/// to the caller.
public func generate(prompt: MLXArray, model: LLMModel, temp: Float = 0.0) -> (
    Task<Void, Never>, AsyncBufferSequence<AsyncChannel<Int>>
) {
    let channel = AsyncChannel<Int>()
    let buffer = channel.buffer(policy: .bounded(10))

    let task = Task {
        var y = prompt
        var cache = [(MLXArray, MLXArray)]()

        while !Task.isCancelled {
            var logits: MLXArray
            (logits, cache) = model(
                expandedDimensions(y, axis: 0), cache: cache.isEmpty ? nil : cache)
            y = sample(logits: logits[-1, axis: 1], temp: temp)
            eval(y)

            await channel.send(y.item(Int.self))
        }
    }

    return (task, buffer)
}
