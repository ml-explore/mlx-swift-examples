// Copyright © 2024 Apple Inc.

import AsyncAlgorithms
import Foundation
import MLX
import MLXRandom
import Tokenizers

private func topPSampling(logits: MLXArray, topP: Float, temp: Float) -> MLXArray {
    var logits = logits
    if logits.dtype == .bfloat16 {
        logits = logits.asType(.float32)
    }

    let probs = softmax(logits / temp, axis: -1)
    let sortedIndices = argSort(probs, axis: -1)

    // probs shape is [B,V] and after take it will be [1, B, V], so we squeeze it back to [B, V]
    let sortedProbs = take(probs, sortedIndices, axis: -1).squeezed(axis: 0)

    let cumulativeProbs = cumsum(sortedProbs, axis: -1)

    let topProbs = MLX.where(cumulativeProbs .> (1 - topP), sortedProbs, zeros(like: sortedProbs))

    let sortedToken = categorical(log(topProbs))
    return sortedIndices.squeezed(axis: 0)[sortedToken]
}

private func applyRepetitionPenalty(
    logits: MLXArray, repetitionContext: MLXArray, penalty: Float
) -> MLXArray {
    if repetitionContext.shape[0] > 0 {
        let indices = repetitionContext
        var selectedLogits = logits[0..., indices]

        selectedLogits = MLX.where(
            selectedLogits .< 0, selectedLogits * penalty, selectedLogits / penalty)

        logits[0..., indices] = selectedLogits
        return logits
    }

    return logits
}

private func sample(logits: MLXArray, temp: Float, topP: Float = 1.0) -> MLXArray {
    if temp == 0 {
        return argMax(logits, axis: -1)
    } else {
        if topP > 0 && topP < 1 {
            return topPSampling(logits: logits, topP: topP, temp: temp)
        }
        return categorical(logits * (1 / temp))
    }
}

/// Parameters for text generation, see ``TokenIterator``
public struct GenerateParameters {
    /// sampling temperature
    public var temperature: Float = 0.6

    /// top p sampling
    public var topP: Float = 1.0

    /// penalty factor for repeating tokens
    public var repetitionPenalty: Float?

    /// number of tokens to consider for repetition penalty
    public var repetitionContextSize: Int = 20

    public init(
        temperature: Float = 0.6, topP: Float = 1.0, repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20
    ) {
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }
}

/// Synchronous generator of tokens.
///
/// Port of `generate_step()` from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py
public struct TokenIterator: Sequence, IteratorProtocol {
    let model: LLMModel
    let parameters: GenerateParameters
    var repetitionContext: MLXArray
    var y: MLXArray
    var cache: [(MLXArray, MLXArray)]

    var first = true

    public init(prompt: MLXArray, model: LLMModel, parameters: GenerateParameters) {
        self.model = model
        self.parameters = parameters
        self.y = prompt
        self.cache = []
        if parameters.repetitionContextSize > 1 {
            if prompt.shape[0] <= parameters.repetitionContextSize {
                self.repetitionContext = prompt
            } else {
                self.repetitionContext = prompt[(-parameters.repetitionContextSize)...]
            }
        } else {
            self.repetitionContext = []
        }
    }

    mutating public func next() -> MLXArray? {
        var logits: MLXArray
        (logits, cache) = model(expandedDimensions(y, axis: 0), cache: cache.isEmpty ? nil : cache)
        logits = logits[0..., -1, 0...]
        if let repetitionPenalty = parameters.repetitionPenalty {
            // apply repetition penalty
            logits = applyRepetitionPenalty(
                logits: logits, repetitionContext: repetitionContext,
                penalty: repetitionPenalty)
        }
        y = sample(logits: logits, temp: parameters.temperature, topP: parameters.topP)
        // append the current token to the context and check repetitionPenalty context see if need to remove the first token
        if parameters.repetitionContextSize > 1 {
            if repetitionContext.shape[0] > parameters.repetitionContextSize {
                repetitionContext = repetitionContext[(-parameters.repetitionContextSize)...]
            }
        }

        return y
    }
}

public struct GenerateResult {
    /// input tokens
    public let promptTokens: [Int]

    /// output tokens
    public let tokens: [Int]

    /// output text
    public let output: String

    /// time to process the prompt / generate the first token
    public let promptTime: TimeInterval

    /// time to generate the remaining tokens
    public let generateTime: TimeInterval

    public var promptTokensPerSecond: Double {
        Double(promptTokens.count) / promptTime
    }

    public var tokensPerSecond: Double {
        Double(tokens.count - 1) / generateTime
    }

    public func summary() -> String {
        """
        Prompt Tokens per second:     \(promptTokensPerSecond.formatted())
        Generation tokens per second: \(tokensPerSecond.formatted())
        """
    }
}

public enum GenerateDisposition {
    case more
    case stop
}

/// Given prompt tokens generate text using the given model and parameters.
///
/// - Parameters:
///   - promptTokens: tokenized prompt
///   - parameters: generation parameters
///   - model: model to evaluate
///   - tokenizer: tokenizer to convert tokens back into strings and recognizer special tokens
///   - configuration: the model configuration
///   - didGenerate: visitor for the tokens as they are generated
public func generate(
    promptTokens: [Int], parameters: GenerateParameters, model: LLMModel, tokenizer: Tokenizer,
    extraEOSTokens: Set<String>? = nil,
    didGenerate: ([Int]) async -> GenerateDisposition
) async -> GenerateResult {
    var start = Date.timeIntervalSinceReferenceDate
    var promptTime: TimeInterval = 0

    let additionalEOSTokenIds = Set(
        (extraEOSTokens ?? [])
            .compactMap {
                tokenizer.convertTokenToId($0)
            })

    var tokens = [Int]()

    for token in TokenIterator(
        prompt: MLXArray(promptTokens), model: model, parameters: parameters)
    {
        // compute the timing for the prompt
        if tokens.isEmpty {
            eval(token)
            let now = Date.timeIntervalSinceReferenceDate
            promptTime = now - start
            start = now
        }

        let t = token.item(Int.self)
        if t == tokenizer.unknownTokenId || t == tokenizer.eosTokenId
            || additionalEOSTokenIds.contains(t)
        {
            break
        }

        tokens.append(t)

        if await didGenerate(tokens) == .stop {
            break
        }
    }

    let now = Date.timeIntervalSinceReferenceDate
    let generateTime = now - start

    return GenerateResult(
        promptTokens: promptTokens, tokens: tokens,
        output: tokenizer.decode(tokens: tokens),
        promptTime: promptTime, generateTime: generateTime)
}
