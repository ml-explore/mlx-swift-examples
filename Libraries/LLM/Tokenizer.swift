// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

/// Wrapper for `Tokenizers.Tokenizer` that provides access to config
/// like ``eosToken``.
public struct Tokenizer: Tokenizers.Tokenizer {

    let tokenizer: Tokenizers.Tokenizer

    public let eosToken: String?
    public let eosTokenId: Int?

    internal init(tokenizer: Tokenizers.Tokenizer, tokenizerConfig: Config) {
        self.tokenizer = tokenizer
        self.eosToken = tokenizerConfig.eosToken?.stringValue
        if let eosToken {
            self.eosTokenId = tokenizer.convertTokenToId(eosToken)
        } else {
            self.eosTokenId = nil
        }
    }

    public func tokenize(text: String) -> [String] {
        tokenizer.tokenize(text: text)
    }

    public func encode(text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    public func decode(tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }

    public func convertTokenToId(_ token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }

    public func convertIdToToken(_ id: Int) -> String? {
        tokenizer.convertIdToToken(id)
    }

    public var unknownToken: String? { tokenizer.unknownToken }

    public var unknownTokenId: Int? { tokenizer.unknownTokenId }

}

public func loadTokenizer(configuration: ModelConfiguration) async throws -> Tokenizer {
    // from AutoTokenizer.from() -- this lets us override parts of the configuration
    let config = LanguageModelConfigurationFromHub(
        modelName: configuration.tokenizerId ?? configuration.id)
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

    let impl = try PreTrainedTokenizer(
        tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)

    return Tokenizer(tokenizer: impl, tokenizerConfig: tokenizerConfig)
}

/// overrides for TokenizerModel/knownTokenizers
let replacementTokenizers = [
    "Qwen2Tokenizer": "PreTrainedTokenizer",
]
