// Copyright © 2024 Apple Inc.

import AsyncAlgorithms
import Foundation
import MLX
import MLXRandom

private func topPSampling(logits: MLXArray, topP: Float, temp: Float) -> MLXArray {
    var logits = logits
    if logits.dtype == .bfloat16 {
        logits = logits.asType(.float32)
    }

    let probs = softMax(logits / temp, axis: -1)
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
    var logits = logits

    if repetitionContext.shape[0] > 0 {
        let indices = repetitionContext
        var selectedLogits = take(logits, indices, axis: -1).squeezed(axis: 0)

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

/// Synchronous generator of tokens.
///
/// Port of `generate_step()` from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py
public struct TokenIterator: Sequence, IteratorProtocol {
    let model: LLMModel
    let temp: Float
    let topP: Float
    let repetitionPenalty: Float
    let repetitionContextSize: Int
    var repetitionContext: MLXArray
    var y: MLXArray
    var cache: [(MLXArray, MLXArray)]

    var first = true

    public init(
        prompt: MLXArray, model: LLMModel, temp: Float = 0.0, topP: Float = 1.0,
        repetitionPenalty: Float = 1.0, repetitionContextSize: Int = 20
    ) {
        self.model = model
        self.temp = temp
        self.topP = topP
        self.y = prompt
        self.cache = []
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        if repetitionContextSize > 1 {
            if prompt.shape[0] <= repetitionContextSize {
                self.repetitionContext = prompt
            } else {
                self.repetitionContext = prompt[-repetitionContextSize ... -1]
            }
        } else {
            self.repetitionContext = []
        }
    }

    mutating public func next() -> MLXArray? {
        var logits: MLXArray
        (logits, cache) = model(expandedDimensions(y, axis: 0), cache: cache.isEmpty ? nil : cache)
        logits = logits[0..., -1, 0...]
        if repetitionPenalty > 1.0 {
            // apply repetition penalty
            logits = applyRepetitionPenalty(
                logits: logits, repetitionContext: repetitionContext, penalty: repetitionPenalty)
        }
        y = sample(logits: logits, temp: temp, topP: topP)
        // append the current token to the context and check repetitionPenalty context see if need to remove the first token
        if repetitionContextSize > 1 {
            repetitionContext = concatenated([repetitionContext, y], axis: 0)
            if repetitionContext.shape[0] > repetitionContextSize {
                repetitionContext = repetitionContext[1...]
            }
        }

        return y
    }
}

/// Async generator of tokens.
///
/// Port of `generate_step()` from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py.
///
/// Note that because MLXArray is not thread safe this eval's the result and sends the TokenId back
/// to the caller.
public func generate(
    prompt: MLXArray, model: LLMModel, temp: Float = 0.0, topP: Float = 1.0,
    repetitionPenalty: Float = 1.0, repetitionContextSize: Int = 20
) -> (
    Task<Void, Never>, AsyncBufferSequence<AsyncChannel<Int>>
) {
    let channel = AsyncChannel<Int>()
    let buffer = channel.buffer(policy: .bounded(10))

    let task = Task {
        var y = prompt
        var cache = [(MLXArray, MLXArray)]()
        var repetitionContext: MLXArray

        if repetitionContextSize > 1 {
            if prompt.shape[0] <= repetitionContextSize {
                repetitionContext = prompt
            } else {
                repetitionContext = prompt[-repetitionContextSize ... -1]
            }
        } else {
            repetitionContext = []
        }
        while !Task.isCancelled {
            var logits: MLXArray
            (logits, cache) = model(
                expandedDimensions(y, axis: 0), cache: cache.isEmpty ? nil : cache)

            logits = logits[0..., -1, 0...]
            if repetitionPenalty > 1.0 {
                // apply repetition penalty
                logits = applyRepetitionPenalty(
                    logits: logits, repetitionContext: repetitionContext, penalty: repetitionPenalty
                )
            }
            y = sample(logits: logits, temp: temp, topP: topP)
            // append the current token to the context and check repetitionPenalty context see if need to remove the first token
            if repetitionContextSize > 1 {
                repetitionContext = concatenated([repetitionContext, y], axis: 0)
                if repetitionContext.shape[0] > repetitionContextSize {
                    repetitionContext = repetitionContext[1...]
                }
            }

            eval(y)

            await channel.send(y.item(Int.self))
        }
    }

    return (task, buffer)
}
