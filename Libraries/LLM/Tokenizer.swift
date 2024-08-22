// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

public func loadTokenizer(configuration: ModelConfiguration, hub: HubApi) async throws -> Tokenizer
{
    let (tokenizerConfig, tokenizerData) = try await loadTokenizerConfig(
        configuration: configuration, hub: hub)

    return try PreTrainedTokenizer(
        tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
}

func loadTokenizerConfig(configuration: ModelConfiguration, hub: HubApi) async throws -> (
    Config, Config
) {
    // from AutoTokenizer.from() -- this lets us override parts of the configuration
    let config: LanguageModelConfigurationFromHub

    switch configuration.id {
    case .id(let id):
        config = LanguageModelConfigurationFromHub(
            modelName: configuration.tokenizerId ?? id, hubApi: hub)
    case .directory(let directory):
        config = LanguageModelConfigurationFromHub(modelFolder: directory, hubApi: hub)
    }

    guard var tokenizerConfig = try await config.tokenizerConfig else {
        throw LLMError(message: "missing config")
    }
    let tokenizerData = try await config.tokenizerData

    tokenizerConfig = updateTokenizerConfig(tokenizerConfig)

    return (tokenizerConfig, tokenizerData)
}

private func updateTokenizerConfig(_ tokenizerConfig: Config) -> Config {
    // workaround: replacement tokenizers for unhandled values in swift-transform
    if let tokenizerClass = tokenizerConfig.tokenizerClass?.stringValue,
        let replacement = replacementTokenizers[tokenizerClass]
    {
        var dictionary = tokenizerConfig.dictionary
        dictionary["tokenizer_class"] = replacement
        return Config(dictionary)
    }

    return tokenizerConfig
}

/// overrides for TokenizerModel/knownTokenizers
let replacementTokenizers = [
    "Qwen2Tokenizer": "PreTrainedTokenizer",
    "CohereTokenizer": "PreTrainedTokenizer",
]
