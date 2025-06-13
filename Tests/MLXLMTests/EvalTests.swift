// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers
import Tokenizers
import XCTest

///
public class EvalTests: XCTestCase {

    func testLlamaEval() throws {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 16, intermediateSize: 512, attentionHeads: 32,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 8)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)

        let input = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 5, 100])
    }

    func testLlamaLora() throws {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 16, intermediateSize: 512, attentionHeads: 32,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 8)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)

        LoRATrain.convert(model: model, layers: model.loraLinearLayers(4))

        let optimizer = Adam(learningRate: 1e-5)

        let train = ["a", "b", "c"]
        let valid = ["x", "y", "z"]

        let tokenizer = TestTokenizer()
        let parameters = LoRATrain.Parameters(iterations: 5)

        try LoRATrain.train(
            model: model, train: train, validate: valid, optimizer: optimizer,
            tokenizer: tokenizer,
            parameters: parameters
        ) { progress in
            print(progress)
            return .more
        }

        let input = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 5, 100])
    }

}

struct TestTokenizer: Tokenizer {

    let length = 8

    var vocabulary: [Int: String]

    init(vocabularySize: Int = 100) {
        let letters = "abcdefghijklmnopqrstuvwxyz"
        self.vocabulary = Dictionary(
            uniqueKeysWithValues: (0 ..< vocabularySize)
                .map { t in
                    (
                        t,
                        String(
                            (0 ..< ((3 ..< 8).randomElement() ?? 3)).compactMap { _ in
                                letters.randomElement()
                            })
                    )
                }
        )
    }

    func tokenize(text: String) -> [String] {
        text.split(separator: " ").map { String($0) }
    }

    func encode(text: String) -> [Int] {
        (0 ..< length).map { _ in
            Int.random(in: 0 ..< 100)
        }
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        encode(text: text)
    }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        var tokens = tokens
        if tokens.count > 50 {
            tokens.append(19)
        }
        return tokens.map { convertIdToToken($0) ?? "" }.joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        Int.random(in: 0 ..< 100)
    }

    func convertIdToToken(_ id: Int) -> String? {
        if id == 19 {
            return "EOS"
        }
        return vocabulary[id]
    }

    var bosToken: String? = nil

    var bosTokenId: Int? = 0

    var eosToken: String? = nil

    var eosTokenId: Int? = 0

    var unknownToken: String? = nil

    var unknownTokenId: Int? = 0

    func applyChatTemplate(messages: [Tokenizers.Message]) throws -> [Int] {
        encode(text: "")
    }

    func applyChatTemplate(messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?) throws
        -> [Int]
    {
        encode(text: "")
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: Any]?
    ) throws -> [Int] {
        encode(text: "")
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument
    ) throws -> [Int] {
        encode(text: "")
    }

    func applyChatTemplate(messages: [Tokenizers.Message], chatTemplate: String) throws -> [Int] {
        encode(text: "")
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?
    ) throws -> [Int] {
        encode(text: "")
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: Any]?
    ) throws -> [Int] {
        encode(text: "")
    }

}
