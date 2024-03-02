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

    // workaround: some merges can't be split on space in BPETokenizer
    if let tokenizerClass = tokenizerConfig.tokenizerClass?.stringValue {
        switch tokenizerClass {
        case "T5Tokenizer":
            break
        default:
            tokenizerData = discardUnhandledMerges(tokenizerData: tokenizerData)
        }
    }

    let impl = try PreTrainedTokenizer(
        tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)

    return Tokenizer(tokenizer: impl, tokenizerConfig: tokenizerConfig)
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
