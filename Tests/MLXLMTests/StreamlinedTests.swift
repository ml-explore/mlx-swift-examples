// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers
import Tokenizers
import XCTest

/// Tests for the streamlined API
public class StreamlinedTests: XCTestCase {

    /// for tests we don't download a model but do execute one
    func model() -> LanguageModel {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 16, intermediateSize: 64, attentionHeads: 8,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 8)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        return model
    }

    /// This is equivalent to:
    ///
    /// ```swift
    /// let model = LLMModelFactory.load("test")
    /// ```
    func modelContainer() -> ModelContainer {
        let context = ModelContext(
            configuration: .init(id: "test", extraEOSTokens: ["EOS"]), model: model(),
            processor: TestUserInputProcessor(), tokenizer: TestTokenizer())
        return ModelContainer(context: context)
    }

    func testOneShot() async throws {
        let model = modelContainer()
        let result = try await ChatSession(model).respond(to: "Tell me about things")
        print(result)
    }

    func testOneShotStream() async throws {
        let model = modelContainer()
        for try await token in ChatSession(model).streamResponse(to: "Tell me about things") {
            print(token, terminator: "")
        }
    }

    func testChat() async throws {
        let model = modelContainer()
        let session = ChatSession(model)

        print(try await session.respond(to: "what color is the sky?"))
        print(try await session.respond(to: "why is that?"))
        print(try await session.respond(to: "describe this image", image: .ciImage(CIImage.red)))
    }
}

private struct TestUserInputProcessor: UserInputProcessor {
    func prepare(input: UserInput) throws -> LMInput {
        LMInput(tokens: MLXRandom.randInt(0 ..< 1000, [100]))
    }
}
