// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

public func loadTokenizer(configuration: ModelConfiguration, hub: HubApi) async throws -> Tokenizer
{
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
    "Qwen2Tokenizer": "PreTrainedTokenizer",
    "CohereTokenizer": "PreTrainedTokenizer",
]

public protocol StreamingDetokenizer: IteratorProtocol<String> {

    mutating func append(token: Int)

}

public struct NaiveStreamingDetokenizer: StreamingDetokenizer {
    let tokenizer: Tokenizer

    var segmentTokens = [Int]()
    var segment = ""

    public init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    mutating public func append(token: Int) {
        segmentTokens.append(token)
    }

    mutating func startNewSegment() {
        let lastToken = segmentTokens.last
        segmentTokens.removeAll()
        if let lastToken {
            segmentTokens.append(lastToken)
            segment = tokenizer.decode(tokens: segmentTokens)
        } else {
            segment = ""
        }
    }

    public mutating func next() -> String? {
        let newSegment = tokenizer.decode(tokens: segmentTokens)
        let new = newSegment.suffix(newSegment.count - segment.count)

        if new.contains("\n") {
            startNewSegment()
        } else {
            self.segment = newSegment
        }

        return String(new)
    }

}
