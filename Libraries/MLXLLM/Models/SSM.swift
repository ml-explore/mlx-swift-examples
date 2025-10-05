//
//  SSM.swift
//  mlx-swift-examples
//
//  Created by Sachin Desai on 10/04/25.
//

import MLX
import MLXNN

/// Compute the clipped time deltas used by SSM-style mixers.
/// - Parameters:
///   - dt: Raw time delta predictions `[batch, seqLen, heads]`.
///   - dtBias: Learned per-head bias `[heads]`.
///   - timeStepLimit: `(min, max)` clipping range.
func computeDt(
    _ dt: MLXArray,
    dtBias: MLXArray,
    timeStepLimit: (Float, Float)
) -> MLXArray {
    let adjusted = dt + dtBias
    let softplusDt = MLXNN.softplus(adjusted)
    return clip(softplusDt, min: timeStepLimit.0, max: timeStepLimit.1)
}

/// Prefix sum helper used by SSM surrogate attention.
/// Expects `x` shaped `[batch, groups, stateDim, seqLen]`.
func segsum(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
    let length = x.dim(-1)
    var values = x

    if let mask {
        let expandedMask = expandedDimensions(mask, axis: 1)
        values = values * expandedMask
    }

    var repeatedValues = expandedDimensions(values, axis: -1)
    repeatedValues = repeated(repeatedValues, count: length, axis: -1)
    repeatedValues = tril(repeatedValues, k: -1)

    var cumulative = cumsum(repeatedValues, axis: -2)

    if let mask {
        let rows = expandedDimensions(mask, axis: -1)
        let cols = expandedDimensions(mask, axis: -2)
        let combined = MLX.logicalAnd(rows, cols)
        let negInf = MLXArray(-Float.infinity, dtype: cumulative.dtype)
        cumulative = MLX.where(combined, cumulative, negInf)
    }

    return cumulative
}

/// Run a single SSM chunk.
func ssmStep(
    dtx: MLXArray,
    dtA: MLXArray,
    B: MLXArray,
    C: MLXArray,
    repeats: Int,
    state: MLXArray?,
    mask: MLXArray?
) -> (MLXArray, MLXArray) {
    let batch = dtx.dim(0)
    let chunk = dtx.dim(1)
    let heads = dtx.dim(2)
    let headDim = dtx.dim(3)
    let groups = B.dim(2)
    let stateDim = B.dim(3)

    var BTransposed = B.transposed(0, 2, 3, 1)
    var CTransposed = C.swappedAxes(1, 2)

    var CB = MLX.matmul(CTransposed, BTransposed)
    CB = repeated(CB, count: repeats, axis: 1)

    let decay = MLX.exp(segsum(dtA.swappedAxes(1, 2), mask: mask))
    let surrogate = tril(CB * decay, k: 0)

    var dtxSwapped = dtx.swappedAxes(1, 2)
    var y = MLX.matmul(surrogate, dtxSwapped)
    y = y.swappedAxes(1, 2)

    let lastIndex = decay.dim(2) - 1
    var decayLast = decay[0..., 0..., lastIndex ..< lastIndex + 1, 0...]
    decayLast = decayLast.transposed(0, 3, 1, 2)

    BTransposed = repeated(BTransposed, count: repeats, axis: 1).swappedAxes(2, 3)

    var dtxDecay = dtx * decayLast
    dtxDecay = dtxDecay.swappedAxes(1, 2).swappedAxes(2, 3)

    var nextState = MLX.matmul(dtxDecay, BTransposed)

    if let state {
        let expDtACumsum = MLX.exp(cumsum(dtA, axis: -2))
        var lastExp = expDtACumsum[0..., (chunk - 1) ..< chunk, 0...]
        lastExp = expandedDimensions(lastExp.squeezed(axis: 1), axis: -1)
        lastExp = expandedDimensions(lastExp, axis: -1)
        nextState = nextState + lastExp * state

        let reshapedState = state.reshaped(
            [batch, 1, groups, repeats, headDim, stateDim])
        let reshapedC = C.reshaped([batch, chunk, groups, 1, stateDim, 1])
        var yPrev = MLX.matmul(reshapedState, reshapedC)
        yPrev = yPrev.squeezed(axis: -1).flattened(start: 2, end: 3)
        let expWeights = expandedDimensions(expDtACumsum, axis: -1)
        y = y + expWeights * yPrev
    }

    return (y, nextState)
}

/// Batched SSM forward pass with optional cache/state.
func ssmUpdate(
    hiddenStates: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray?,
    timeStepLimit: (Float, Float),
    mask: MLXArray?,
    chunkSize: Int = 256
) -> (MLXArray, MLXArray) {
    let batch = hiddenStates.dim(0)
    let length = hiddenStates.dim(1)
    let heads = hiddenStates.dim(2)
    let headDim = hiddenStates.dim(3)
    let groups = B.dim(2)

    let repeats = heads / groups

    let dtClipped = computeDt(dt, dtBias: dtBias, timeStepLimit: timeStepLimit)
    let A = -MLX.exp(ALog)
    let dtA = dtClipped * A.reshaped([1, 1, -1])
    let dtx = dtClipped.reshaped([batch, length, heads, 1]) * hiddenStates

    var outputs: [MLXArray] = []
    var currentState = state

    for start in stride(from: 0, to: length, by: chunkSize) {
        let end = min(start + chunkSize, length)
        let range = start ..< end

        let dtxSlice = dtx[0..., range, 0..., 0...]
        let dtASlice = dtA[0..., range, 0...]
        let bSlice = B[0..., range, 0..., 0...]
        let cSlice = C[0..., range, 0..., 0...]
        let maskSlice = mask?[0..., range]

        let (chunkOut, nextState) = ssmStep(
            dtx: dtxSlice,
            dtA: dtASlice,
            B: bSlice,
            C: cSlice,
            repeats: repeats,
            state: currentState,
            mask: maskSlice
        )
        outputs.append(chunkOut)
        currentState = nextState
    }

    var y = concatenated(outputs, axis: 1)
    y = y + hiddenStates * D.reshaped([1, 1, heads, 1])

    let stateDim = B.dim(3)
    let finalState =
        currentState
        ?? MLXArray.zeros([batch, heads, headDim, stateDim], dtype: hiddenStates.dtype)
    return (y, finalState)
}
