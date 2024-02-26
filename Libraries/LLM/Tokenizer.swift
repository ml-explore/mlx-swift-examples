// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

public func loadTokenizer(name: String) async throws -> Tokenizer {
    // from AutoTokenizer.from() -- this lets us override parts of the configuration
    let config = LanguageModelConfigurationFromHub(modelName: name)
    guard var tokenizerConfig = try await config.tokenizerConfig else {
        throw LLMError(message: "missing config")
    }
    var tokenizerData = try await config.tokenizerData

    // workaround: replacement tokenizers for unhandled values in swift-transform
    if let tokenizerClass = tokenizerConfig.tokenizerClass?.stringValue,
        let replacement = replacementTokenizers[tokenizerClass]
    {
        var dictionary = tokenizerConfig.dictionary
        dictionary["tokenizer_class"] = replacement
        tokenizerConfig = Config(dictionary)
    }

    // workaround: some merges can't be split on space in BPETokenizer
    if let tokenizerClass = tokenizerConfig.tokenizerClass?.stringValue {
        switch tokenizerClass {
        case "T5Tokenizer":
            break
        default:
            tokenizerData = discardUnhandledMerges(tokenizerData: tokenizerData)
        }
    }

    return try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
}

public func discardUnhandledMerges(tokenizerData: Config) -> Config {
    // see https://github.com/ml-explore/mlx-swift-examples/issues/1
    // and https://github.com/huggingface/swift-transformers/issues/51

    if let model = tokenizerData.model {
        if let merges = model.dictionary["merges"] as? [String] {
            // discard any merges that can't be split on a space
            // (required by BPETokenizer)
            let newMerges =
                merges
                .filter {
                    $0.split(separator: " ").count == 2
                }

            if newMerges.count != merges.count {
                var newModel = model.dictionary
                newModel["merges"] = newMerges

                var newTokenizerData = tokenizerData.dictionary
                newTokenizerData["model"] = newModel

                return Config(newTokenizerData)
            }
        }
    }

    return tokenizerData
}

/// overrides for TokenizerModel/knownTokenizers
let replacementTokenizers = [
    "CodeLlamaTokenizer": "LlamaTokenizer",
    "GemmaTokenizer": "PreTrainedTokenizer",
]
