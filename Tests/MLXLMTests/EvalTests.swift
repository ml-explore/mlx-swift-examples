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

        let adapter = try LoRAContainer.from(
            model: model,
            configuration: LoRAConfiguration(numLayers: 4)
        )

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

    func testConcurrentEvaluation() async throws {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)

        // Force evaluation of all model weights before concurrent usage
        // This ensures all weight promises are realized and avoids race conditions
        eval(model)

        let numTasks = 3
        let results = await withTaskGroup(of: MLXArray.self) { group in
            var allResults: [MLXArray] = []

            for taskId in 0 ..< numTasks {
                group.addTask {
                    let input = MLXArray([
                        1 + taskId, 2 + taskId, 3 + taskId, 4 + taskId, 5 + taskId,
                    ])[.newAxis, .ellipsis]
                    let output = model.callAsFunction(input, cache: nil)
                    return output
                }
            }

            for await result in group {
                allResults.append(result)
            }

            return allResults
        }

        XCTAssertEqual(results.count, numTasks)

        for (index, result) in results.enumerated() {
            XCTAssertEqual(result.shape, [1, 5, 100])
        }
    }

    func testConcurrentSampling() async throws {
        let vocabSize = 100
        let logits = MLXRandom.normal([1, vocabSize])

        let numSamplers = 4
        let results = try await withThrowingTaskGroup(of: Int.self) { group in
            var samplerResults: [Int] = []

            for samplerId in 0 ..< numSamplers {
                group.addTask {
                    return try withRandomState(MLXRandom.RandomState(seed: UInt64(samplerId))) {
                        if samplerId % 2 == 0 {
                            return categorical(logits).item(Int.self)
                        } else {
                            return logits.argMax(axis: -1).item(Int.self)
                        }
                    }
                }
            }

            for try await result in group {
                samplerResults.append(result)
            }

            return samplerResults
        }

        XCTAssertEqual(results.count, numSamplers)

        for result in results {
            XCTAssertGreaterThanOrEqual(result, 0)
            XCTAssertLessThan(result, vocabSize)
        }
    }

    func testRandomStateIsolation() async throws {
        let config = LlamaConfiguration(
            hiddenSize: 32, hiddenLayers: 2, intermediateSize: 64, attentionHeads: 4,
            rmsNormEps: 0.00001, vocabularySize: 50, kvHeads: 2)

        // Force evaluation of all model weights before concurrent usage
        // This ensures all weight promises are realized and avoids race conditions
        let model = LlamaModel(config)
        eval(model)

        let sharedLogits = MLXArray.ones([1, 50])
        let numSamplers = 5
        let samplesPerTask = 10

        let allResults = try await withThrowingTaskGroup(of: [Int].self) { group in
            var results: [[Int]] = []

            for samplerId in 0 ..< numSamplers {
                group.addTask {
                    var taskResults: [Int] = []
                    let sampler = CategoricalSampler(temperature: 1.0)

                    for sampleId in 0 ..< samplesPerTask {
                        let token = try withRandomState(
                            MLXRandom.RandomState(seed: UInt64(samplerId * 1000 + sampleId))
                        ) {
                            return sampler.sample(logits: sharedLogits)
                        }
                        taskResults.append(token.item(Int.self))
                    }

                    return taskResults
                }
            }

            for try await result in group {
                results.append(result)
            }

            return results
        }

        XCTAssertEqual(allResults.count, numSamplers)

        for samplerResults in allResults {
            XCTAssertEqual(samplerResults.count, samplesPerTask)
        }

        let uniqueSequences = Set(allResults.map { $0.description })
        XCTAssertGreaterThan(uniqueSequences.count, 0)
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
