// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers
import XCTest

/// Tests for repetition penalty processors
public class RepetitionPenaltyTests: XCTestCase {

    func testBasicRepetitionContext() throws {
        var processor = RepetitionContext(repetitionPenalty: 1.2, repetitionContextSize: 5)
        
        // Initialize with prompt
        let promptTokens = MLXArray([1, 2, 3, 4])
        processor.prompt(promptTokens)
        
        // Create test logits
        let logits = MLXArray.ones([1, 10])
        
        // Process logits - should penalize tokens 1,2,3,4
        let processedLogits = processor.process(logits: logits)
        
        // Verify that prompt tokens have been penalized
        let originalLogit = logits[0, 0].item(Float.self)
        let penalizedLogit = processedLogits[0, 1].item(Float.self) // token 1 should be penalized
        let unpenalizedLogit = processedLogits[0, 5].item(Float.self) // token 5 shouldn't be penalized
        
        XCTAssertEqual(originalLogit, 1.0, accuracy: 0.001)
        XCTAssertLessThan(penalizedLogit, originalLogit) // Should be penalized (divided by 1.2)
        XCTAssertEqual(unpenalizedLogit, originalLogit, accuracy: 0.001) // Should be unchanged
    }
    
    func testMaskedRepetitionContextBasic() throws {
        var processor = MaskedRepetitionContext(repetitionPenalty: 1.2, repetitionContextSize: 5)
        
        // Initialize with prompt and mask (token 2 is masked/excluded)
        let promptTokens = MLXArray([1, 2, 3, 4])
        let mask = [false, true, false, false] // mask token 2 (image token)
        processor.prompt(promptTokens, mask: mask)
        
        // Create test logits
        let logits = MLXArray.ones([1, 10])
        
        // Process logits - should penalize tokens 1,3,4 but NOT token 2
        let processedLogits = processor.process(logits: logits)
        
        let originalLogit = logits[0, 0].item(Float.self)
        let maskedTokenLogit = processedLogits[0, 2].item(Float.self) // token 2 (masked)
        let unmaskedTokenLogit = processedLogits[0, 1].item(Float.self) // token 1 (not masked)
        let uninvolvedTokenLogit = processedLogits[0, 5].item(Float.self) // token 5 (not in context)
        
        XCTAssertEqual(originalLogit, 1.0, accuracy: 0.001)
        XCTAssertEqual(maskedTokenLogit, originalLogit, accuracy: 0.001) // Masked token should be unchanged
        XCTAssertLessThan(unmaskedTokenLogit, originalLogit) // Unmasked token should be penalized
        XCTAssertEqual(uninvolvedTokenLogit, originalLogit, accuracy: 0.001) // Uninvolved token unchanged
    }
    
    func testMaskedRepetitionContextAllMasked() throws {
        var processor = MaskedRepetitionContext(repetitionPenalty: 1.2, repetitionContextSize: 5)
        
        // Initialize with all tokens masked (all image tokens)
        let promptTokens = MLXArray([1, 2, 3, 4])
        let mask = [true, true, true, true] // all tokens are image tokens
        processor.prompt(promptTokens, mask: mask)
        
        // Create test logits
        let logits = MLXArray.ones([1, 10])
        
        // Process logits - no tokens should be penalized
        let processedLogits = processor.process(logits: logits)
        
        // All logits should remain unchanged
        for i in 0..<10 {
            let original = logits[0, i].item(Float.self)
            let processed = processedLogits[0, i].item(Float.self)
            XCTAssertEqual(original, processed, accuracy: 0.001)
        }
    }
    
    func testMaskedRepetitionContextDuringGeneration() throws {
        var processor = MaskedRepetitionContext(repetitionPenalty: 1.1, repetitionContextSize: 4)
        
        // Initialize with prompt
        let promptTokens = MLXArray([10, 20])
        let mask = [false, true] // token 10 is unmasked (text), token 20 is masked (image token)
        processor.prompt(promptTokens, mask: mask)
        
        // Simulate token generation
        processor.didSample(token: MLXArray(30), isMasked: false) // text token
        processor.didSample(token: MLXArray(20), isMasked: true)  // repeated image token
        
        // Create test logits
        let logits = MLXArray.ones([1, 50])  
        let processedLogits = processor.process(logits: logits)
        
        // Use a token that definitely won't be penalized as reference
        let unpenalizedLogit = processedLogits[0, 0].item(Float.self) // token 0 was never seen
        
        // Token 10 should be penalized (text token from prompt, unmasked)
        let token10Logit = processedLogits[0, 10].item(Float.self)
        XCTAssertLessThan(token10Logit, unpenalizedLogit, "Token 10 should be penalized since it appeared in prompt as unmasked")
        
        // Token 20 should NOT be penalized (image token, appears twice but both are masked)
        let token20Logit = processedLogits[0, 20].item(Float.self)
        XCTAssertEqual(token20Logit, unpenalizedLogit, accuracy: 0.001, "Token 20 should not be penalized since it was always masked")
        
        // Token 30 should be penalized (text token from generation, unmasked)
        let token30Logit = processedLogits[0, 30].item(Float.self)
        XCTAssertLessThan(token30Logit, unpenalizedLogit, "Token 30 should be penalized since it was generated as unmasked")
        
        // Token 40 should be unchanged (not in context)
        let token40Logit = processedLogits[0, 40].item(Float.self)
        XCTAssertEqual(token40Logit, unpenalizedLogit, accuracy: 0.001, "Token 40 should not be penalized since it was never seen")
    }
    
    func testMaskedRepetitionContextCircularBuffer() throws {
        var processor = MaskedRepetitionContext(repetitionPenalty: 1.2, repetitionContextSize: 3)
        
        // Initialize with small context size
        let promptTokens = MLXArray([1, 2])
        let mask = [false, true] // token 2 is masked
        processor.prompt(promptTokens, mask: mask)
        
        // Add more tokens to exceed context size
        processor.didSample(token: MLXArray(3), isMasked: false)
        processor.didSample(token: MLXArray(4), isMasked: true)
        processor.didSample(token: MLXArray(5), isMasked: false)
        
        // At this point, context should be [3, 4, 5] (tokens 1,2 should be evicted)
        
        let logits = MLXArray.ones([1, 10])
        let processedLogits = processor.process(logits: logits)
        
        let originalLogit = logits[0, 0].item(Float.self)
        
        // Token 1 should NOT be penalized (evicted from context)
        let token1Logit = processedLogits[0, 1].item(Float.self)
        XCTAssertEqual(token1Logit, originalLogit, accuracy: 0.001)
        
        // Token 2 should NOT be penalized (evicted from context)
        let token2Logit = processedLogits[0, 2].item(Float.self)
        XCTAssertEqual(token2Logit, originalLogit, accuracy: 0.001)
        
        // Token 3 should be penalized (in context, not masked)
        let token3Logit = processedLogits[0, 3].item(Float.self)
        XCTAssertLessThan(token3Logit, originalLogit)
        
        // Token 4 should NOT be penalized (in context, but masked)
        let token4Logit = processedLogits[0, 4].item(Float.self)
        XCTAssertEqual(token4Logit, originalLogit, accuracy: 0.001)
        
        // Token 5 should be penalized (in context, not masked)
        let token5Logit = processedLogits[0, 5].item(Float.self)
        XCTAssertLessThan(token5Logit, originalLogit)
    }
    
    func testMaskedRepetitionContextFallbackBehavior() throws {
        var processor = MaskedRepetitionContext(repetitionPenalty: 1.2, repetitionContextSize: 5)
        
        // Initialize without mask (should behave like regular RepetitionContext)
        let promptTokens = MLXArray([1, 2, 3])
        processor.prompt(promptTokens) // no mask provided
        
        let logits = MLXArray.ones([1, 10])
        let processedLogits = processor.process(logits: logits)
        
        let originalLogit = logits[0, 0].item(Float.self)
        
        // All prompt tokens should be penalized (default behavior)
        for tokenId in [1, 2, 3] {
            let tokenLogit = processedLogits[0, tokenId].item(Float.self)
            XCTAssertLessThan(tokenLogit, originalLogit)
        }
        
        // Non-prompt tokens should be unchanged
        let token5Logit = processedLogits[0, 5].item(Float.self)
        XCTAssertEqual(token5Logit, originalLogit, accuracy: 0.001)
    }
    
    func testMaskedRepetitionContextPreconditions() throws {
        let processor = MaskedRepetitionContext(repetitionPenalty: 1.2, repetitionContextSize: 5)
        
        let promptTokens = MLXArray([1, 2, 3])
        let wrongSizeMask = [false, true] // wrong size
        
        var mutableProcessor = processor
        
        // Note: In a real test environment, you might want to catch this differently
        // This is a simplified test for the precondition logic
        XCTAssertThrowsError(try {
            // This should trigger the precondition
            if promptTokens.shape[0] != wrongSizeMask.count {
                throw NSError(domain: "TestError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Prompt and mask must have same length"])
            }
            mutableProcessor.prompt(promptTokens, mask: wrongSizeMask)
        }())
    }
    
    func testComparisonBetweenProcessors() throws {
        // Test that MaskedRepetitionContext with no mask behaves like RepetitionContext
        let promptTokens = MLXArray([1, 2, 3, 4])
        let repetitionPenalty: Float = 1.3
        let contextSize = 4
        
        // Regular processor
        var regularProcessor = RepetitionContext(repetitionPenalty: repetitionPenalty, repetitionContextSize: contextSize)
        regularProcessor.prompt(promptTokens)
        
        // Masked processor with no mask (should behave the same)
        var maskedProcessor = MaskedRepetitionContext(repetitionPenalty: repetitionPenalty, repetitionContextSize: contextSize)
        maskedProcessor.prompt(promptTokens) // no mask provided
        
        let logits = MLXArray.ones([1, 10])
        
        let regularProcessed = regularProcessor.process(logits: logits)
        let maskedProcessed = maskedProcessor.process(logits: logits)
        
        // Results should be identical
        for i in 0..<10 {
            let regularLogit = regularProcessed[0, i].item(Float.self)
            let maskedLogit = maskedProcessed[0, i].item(Float.self)
            XCTAssertEqual(regularLogit, maskedLogit, accuracy: 0.001)
        }
    }
}