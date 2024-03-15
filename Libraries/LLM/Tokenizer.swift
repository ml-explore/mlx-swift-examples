// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

public func loadTokenizer(configuration: ModelConfiguration) async throws -> Tokenizer {
    // from AutoTokenizer.from() -- this lets us override parts of the configuration
    let config = LanguageModelConfigurationFromHub(
        modelName: configuration.tokenizerId ?? configuration.id)
    guard var tokenizerConfig = try await config.tokenizerConfig else {
        throw LLMError(message: "missing config")
    }
    let tokenizerData = try await config.tokenizerData

    // workaround: replacement tokenizers for unhandled values in swift-transform
    if let tokenizerClass = tokenizerConfig.tokenizerClass?.stringValue,
        let replacement = replacementTokenizers[tokenizerClass]
    {
        var dictionary = tokenizerConfig.dictionary
        dictionary["tokenizer_class"] = replacement
        tokenizerConfig = Config(dictionary)
    }

    return try PreTrainedTokenizer(
        tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
}

/// overrides for TokenizerModel/knownTokenizers
let replacementTokenizers = [
    "Qwen2Tokenizer": "PreTrainedTokenizer"
]
