// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXRandom
import Tokenizers

/// Parameters for text generation, see ``TokenIterator``
public struct GenerateParameters: Sendable {

    /// Step size for processing the prompt
    public var prefillStepSize = 512

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

struct SampleContext {

    let temp: MLXArray
    let topP: MLXArray
    let useTopP: Bool
    let useArgMax: Bool

    init(parameters: GenerateParameters) {
        self.temp = MLXArray(parameters.temperature)
        self.topP = MLXArray(parameters.topP)
        self.useTopP = parameters.topP > 0 && parameters.topP < 1
        self.useArgMax = parameters.temperature == 0
    }

    private let compiledTopPSampling: (MLXArray, MLXArray, MLXArray) -> MLXArray = {
        compile(inputs: [MLXRandom.globalState], outputs: [MLXRandom.globalState]) {
            logits, topP, temp in
            let probs = softmax(logits / temp, axis: -1)
            let sortedIndices = argSort(probs, axis: -1)

            // probs shape is [B,V] and after take it will be [1, B, V], so we squeeze it back to [B, V]
            let sortedProbs = take(probs, sortedIndices, axis: -1).squeezed(axis: 0)

            let cumulativeProbs = cumsum(sortedProbs, axis: -1)

            let topProbs = MLX.where(
                cumulativeProbs .> (1 - topP), sortedProbs, zeros(like: sortedProbs))

            let sortedToken = categorical(log(topProbs))
            return sortedIndices.squeezed(axis: 0)[sortedToken]
        }
    }()

    private let compiledCategorical: (MLXArray, MLXArray) -> MLXArray = {
        compile(inputs: [MLXRandom.globalState], outputs: [MLXRandom.globalState]) { logits, temp in
            categorical(logits * (1 / temp))
        }
    }()

    private func topPSampling(logits: MLXArray) -> MLXArray {
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        return compiledTopPSampling(logits, topP, temp)
    }

    func sample(logits: MLXArray) -> MLXArray {
        if useArgMax {
            return argMax(logits, axis: -1)
        } else {
            if useTopP {
                return topPSampling(logits: logits)
            } else {
                return compiledCategorical(logits, temp)
            }
        }
    }
}

/// Encapsulaton of the repetitionPenalty
struct RepetitionContext: Sendable {
    /// tokens in the repetition context sliding window
    var tokens: [Int]

    /// current write into into the tokens circular array
    var index = 0

    /// penalty factor for repeating tokens
    let repetitionPenalty: Float?

    /// number of tokens to consider for repetition penalty
    let repetitionContextSize: Int

    init(prompt: MLXArray, parameters: GenerateParameters) {
        self.repetitionPenalty = parameters.repetitionPenalty
        self.repetitionContextSize = parameters.repetitionContextSize

        if repetitionPenalty != nil && repetitionContextSize > 1 {
            if prompt.shape[0] <= repetitionContextSize {
                self.tokens = prompt.asArray(Int.self)
            } else {
                self.tokens = prompt[(-repetitionContextSize)...].asArray(Int.self)
            }
        } else {
            self.tokens = []
        }
    }

    func applyRepetitionPenalty(logits: MLXArray) -> MLXArray {
        if let penalty = repetitionPenalty, tokens.count > 0 {
            let indices = MLXArray(tokens.map { UInt32($0) })
            var selectedLogits = logits[0..., indices]

            selectedLogits = MLX.where(
                selectedLogits .< 0, selectedLogits * penalty, selectedLogits / penalty)

            logits[0..., indices] = selectedLogits
            return logits
        }

        return logits
    }

    mutating func append(token: MLXArray) {
        if repetitionPenalty != nil {
            if tokens.count >= repetitionContextSize {
                tokens[index] = token.item(Int.self)
                index = (index + 1) % repetitionContextSize
            } else {
                tokens.append(token.item(Int.self))
            }
        }
    }
}

/// Synchronous generator of tokens.
///
/// Tokens are integers that can be passed through a `Tokenizer` or ``StreamingDetokenizer`` to produce Strings.
///
/// Port of `generate_step()` from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py
///
/// Note: this uses `asyncEval()` and there may be an async evaluation running after a call to `next()`.
public struct TokenIterator: Sequence, IteratorProtocol {
    let model: LLMModel
    let parameters: GenerateParameters

    var y: MLXArray
    var cache: [KVCache]
    var repetitionContext: RepetitionContext
    let sampleContext: SampleContext

    public init(prompt: MLXArray, model: LLMModel, parameters: GenerateParameters) {
        self.model = model
        self.parameters = parameters
        self.y = prompt
        self.cache = model.newCache(parameters: parameters)

        self.repetitionContext = RepetitionContext(prompt: prompt, parameters: parameters)
        self.sampleContext = SampleContext(parameters: parameters)

        // prepare the prompt in chunks if larger than the prefill size
        while y.size > parameters.prefillStepSize {
            _ = model(
                y[.newAxis, ..<parameters.prefillStepSize], cache: cache.isEmpty ? nil : cache)
            eval(cache)
            y = y[parameters.prefillStepSize...]
        }

        // evaluate the remainder of the prompt -- this primes the pump
        y = step(previous: y)
        asyncEval(y)
    }

    /// Evaluate the next token and return the new token (y) and cache state.
    ///
    /// This may mutate the repititionContext.
    mutating func step(previous: MLXArray) -> MLXArray {
        var logits: MLXArray
        logits = model(previous[.newAxis], cache: cache.isEmpty ? nil : cache)

        logits = logits[0..., -1, 0...]
        logits = repetitionContext.applyRepetitionPenalty(logits: logits)

        let y = sampleContext.sample(logits: logits)

        repetitionContext.append(token: y)

        return y
    }

    mutating public func next() -> Int? {
        // save current value -- this will be returned
        let previousY = y

        // compute the next state and async eval the next token
        y = step(previous: previousY)
        asyncEval(y)

        return previousY.item(Int.self)
    }
}

public struct GenerateResult: Sendable {
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
        Double(tokens.count) / generateTime
    }

    public func summary() -> String {
        """
        Prompt:     \(promptTokens.count) tokens, \(promptTokensPerSecond.formatted()) tokens/s
        Generation: \(tokens.count) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
    }
}

public enum GenerateDisposition: Sendable {
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
    didGenerate: ([Int]) -> GenerateDisposition
) -> GenerateResult {
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
            let now = Date.timeIntervalSinceReferenceDate
            promptTime = now - start
            start = now
        }

        if token == tokenizer.unknownTokenId || token == tokenizer.eosTokenId
            || additionalEOSTokenIds.contains(token)
        {
            break
        }
        tokens.append(token)

        if didGenerate(tokens) == .stop {
            break
        }
    }

    let now = Date.timeIntervalSinceReferenceDate
    let generateTime = now - start

    // TokenIterator uses `asyncEval()` to keep the pipeline full.  If the caller
    // exits the program right away, those tasks will still be executing and will
    // hit assertions as the mlx scheduler is torn down.  Synchronize with the stream
    // to make sure it is complete.
    Stream().synchronize()

    return GenerateResult(
        promptTokens: promptTokens, tokens: tokens,
        output: tokenizer.decode(tokens: tokens),
        promptTime: promptTime, generateTime: generateTime)
}
