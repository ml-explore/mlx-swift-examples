//
//  FalconH1.swift
//  mlx-swift-examples
//
//  Created by John Mai on 2025/6/18.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/falcon_h1.py

// MARK: - RMSNormGated

private class RMSNormGated: Module {
    let weight: MLXArray
    let varianceEpsilon: Float
    let nGroups: Int
    let normBeforeGate: Bool

    init(hiddenSize: Int, eps: Float = 1e-6, nGroups: Int = 1, normBeforeGate: Bool = true) {
        self.weight = MLXArray.ones([hiddenSize])
        self.varianceEpsilon = eps
        self.nGroups = nGroups
        self.normBeforeGate = normBeforeGate
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
        let inputDtype = hiddenStates.dtype

        var hiddenStates = hiddenStates

        if !normBeforeGate, let gate {
            hiddenStates = hiddenStates * silu(gate.asType(.float16))
        }

        hiddenStates = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: varianceEpsilon)

        if normBeforeGate, let gate {
            hiddenStates = hiddenStates * silu(gate.asType(.float16))
        }

        return hiddenStates.asType(inputDtype)
    }
}

private func computeMupVector(_ args: FalconH1Configuration) -> MLXArray {
    let intermediateSize = args.mambaDSSM ?? Int(Float(args.mambaExpand) * Float(args.hiddenSize))
    let groupsTimeStateSize = args.mambaNGroups * args.mambaDState
    let numHeads = args.mambaNHeads
    let zxbcdtMultipliers = args.ssmMultipliers

    let vectorShape = 2 * intermediateSize + 2 * groupsTimeStateSize + numHeads
    let mupVector = MLXArray.ones([1, 1, vectorShape])

    mupVector[0..., 0..., ..<intermediateSize] *= zxbcdtMultipliers[0]
    mupVector[0..., 0..., intermediateSize ..< (2 * intermediateSize)] *= zxbcdtMultipliers[1]
    mupVector[
        0..., 0..., (2 * intermediateSize) ..< (2 * intermediateSize + groupsTimeStateSize)] *=
        zxbcdtMultipliers[2]
    mupVector[
        0..., 0...,
        (2 * intermediateSize + groupsTimeStateSize)
            ..< (2 * intermediateSize + 2 * groupsTimeStateSize)] *= zxbcdtMultipliers[3]
    mupVector[0..., 0..., (2 * intermediateSize + 2 * groupsTimeStateSize)...] *=
        zxbcdtMultipliers[4]

    return mupVector
}

// MARK: - Attention

private class Attention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let layerIdx: Int
    let keyMultiplier: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(_ args: FalconH1Configuration, layerIdx: Int) {
        self.hiddenSize = args.hiddenSize
        self.numHeads = args.numAttentionHeads
        self.numKVHeads = args.numKeyValueHeads
        self.headDim = args.headDim ?? args.hiddenSize / args.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)
        self.layerIdx = layerIdx
        self.keyMultiplier = args.keyMultiplier

        _qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: args.attentionBias)

        let ropeScale: Float =
            if let ropeScaling = args.ropeScaling {
                1 / ropeScaling
            } else {
                1
            }

        self.rope = RoPE(
            dimensions: headDim,
            traditional: args.ropeTraditional,
            base: args.ropeTheta,
            scale: ropeScale
        )
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: Mamba2Cache? = nil) -> MLXArray
    {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        keys = keys * keyMultiplier

        queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.seqlenOffset)
            keys = rope(keys, offset: cache.seqlenOffset)
            (keys, values) = cache.update(keyStates: keys, valueStates: values, layerIdx: layerIdx)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        if var mask {
            let kvSeqLen = keys.dim(2)
            if mask.ndim == 2 {
                mask = mask[.newAxis, .newAxis, 0..., 0...]
            }

            if kvSeqLen > L {
                if mask.dim(-1) < kvSeqLen {
                    let numHeadsDim = mask.dim(1) > 1 ? mask.dim(1) : 1
                    let padLength = kvSeqLen - mask.dim(-1)
                    let padShape = [B, numHeadsDim, L, padLength]
                    let padding = MLXArray.ones(padShape, dtype: mask.dtype)
                    mask = concatenated([padding, mask], axis: -1)
                }
            }
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(output)
    }
}

// MARK: - MLP

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let gateMultiplier: Float
    let downMultiplier: Float

    init(_ args: FalconH1Configuration) {
        let hiddenSize = args.hiddenSize
        let intermediateSize = args.intermediateSize

        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: args.mlpBias)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: args.mlpBias)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: args.mlpBias)

        self.gateMultiplier = args.mlpMultipliers[0]
        self.downMultiplier = args.mlpMultipliers[1]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = upProj(x) * silu(gateProj(x) * gateMultiplier)
        return downProj(y) * downMultiplier
    }
}

// MARK: - DecoderLayer

private class DecoderLayer: Module {
    let mamba: Mixer
    let channelsAttn: Int
    let ssmOutMultiplier: Float
    let attnOutMultiplier: Float
    let attentionInMultiplier: Float

    @ModuleInfo(key: "feed_forward") var feedForward: MLP
    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_ff_layernorm") var preFfLayerNorm: RMSNorm

    init(_ args: FalconH1Configuration, layerIdx: Int, mupVector: MLXArray) {
        self.mamba = Mixer(args, layerIdx: layerIdx, mupVector: mupVector)

        let headDim = args.hiddenSize / args.numAttentionHeads
        self.channelsAttn = args.numAttentionHeads * headDim + 2 * args.numKeyValueHeads * headDim

        self.attentionInMultiplier = args.attentionInMultiplier
        self.ssmOutMultiplier = args.ssmOutMultiplier
        self.attnOutMultiplier = args.attentionOutMultiplier

        _feedForward.wrappedValue = MLP(args)
        _attention.wrappedValue = Attention(args, layerIdx: layerIdx)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFfLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        cache: Mamba2Cache,
        mask: MLXArray?,
        mambaMask: MLXArray?,
        cachePosition: MLXArray
    ) -> MLXArray {
        var residual = hiddenStates
        var hiddenStates = inputLayerNorm(hiddenStates)

        let mambaHiddenStates =
            mamba(
                hiddenStates,
                cache: cache,
                mask: mambaMask,
                cachePosition: cachePosition
            ) * ssmOutMultiplier

        let attentionHiddenStates =
            attention(
                hiddenStates * attentionInMultiplier,
                mask: mask,
                cache: cache
            ) * attnOutMultiplier

        hiddenStates = mambaHiddenStates + attentionHiddenStates

        hiddenStates = residual + hiddenStates

        residual = hiddenStates
        hiddenStates = preFfLayerNorm(hiddenStates)
        hiddenStates = feedForward(hiddenStates)
        hiddenStates = residual + hiddenStates

        return hiddenStates
    }
}

private func applyMaskToPaddingStates(_ inputStates: MLXArray, _ attentionMask: MLXArray?)
    -> MLXArray
{
    if let attentionMask {
        let mask = expandedDimensions(attentionMask, axes: [-1])
        return inputStates * mask
    }
    return inputStates
}

private func padTensorBySize(_ tensor: MLXArray, _ padSize: Int) -> MLXArray {
    if padSize > 0 {
        var padShape = tensor.shape
        padShape[1] = padSize
        let padding = MLXArray.zeros(padShape).asType(tensor.dtype)
        return concatenated([tensor, padding], axis: 1)
    }
    return tensor
}

private func reshapeIntoChunks(_ tensor: MLXArray, _ padSize: Int, _ chunkSize: Int) -> MLXArray {
    var tensor = tensor
    if padSize > 0 {
        tensor = padTensorBySize(tensor, padSize)
    }

    let batchSize = tensor.shape[0]
    let seqLen = tensor.dim(1)
    let numChunks = seqLen / chunkSize

    var newShape = [batchSize, numChunks, chunkSize]
    newShape.append(contentsOf: Array(tensor.shape[2...]))
    return tensor.reshaped(newShape)
}

private func segmentSum(_ inputTensor: MLXArray) -> MLXArray {
    let chunkSize = inputTensor.dim(-1)
    var inputTensor = expandedDimensions(inputTensor, axes: [-1])
    inputTensor = broadcast(
        inputTensor, to: inputTensor.shape[0 ..< inputTensor.ndim - 1] + [chunkSize])

    var mask = tri(chunkSize, k: -1).asType(.bool)
    inputTensor = MLX.where(mask, inputTensor, MLXArray.zeros(like: inputTensor))

    let tensorSegsum = cumsum(inputTensor, axis: -2)

    mask = tri(chunkSize, k: 0).asType(.bool)
    return MLX.where(mask, tensorSegsum, MLXArray(-Float.infinity))
}

// MARK: - Mixer

private class Mixer: Module {
    let numHeads: Int
    let hiddenSize: Int
    let ssmStateSize: Int
    let convKernelSize: Int
    let intermediateSize: Int
    let layerIdx: Int
    let useConvBias: Bool
    let useBias: Bool
    let layerNormEpsilon: Float
    let groupsTimeStateSize: Int
    let nGroups: Int
    let headDim: Int
    let chunkSize: Int
    let timeStepLimit: (Float, Float)
    let timeStepMin: Float
    let timeStepMax: Float
    let convDim: Int
    let mambaRMSNorm: Bool
    let norm: RMSNormGated?
    let ssmInMultiplier: Float
    let conv1d: Conv1d

    let _mupVector: MLXArray

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var d: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: FalconH1Configuration, layerIdx: Int, mupVector: MLXArray) {
        self.numHeads = args.mambaNHeads
        self.hiddenSize = args.hiddenSize
        self.ssmStateSize = args.mambaDState
        self.convKernelSize = args.mambaDConv
        self.intermediateSize = args.mambaDSSM ?? Int(args.mambaExpand * args.hiddenSize)
        self.layerIdx = layerIdx
        self.useConvBias = args.mambaConvBias
        self.useBias = args.mambaProjBias
        self.layerNormEpsilon = args.rmsNormEps
        self.groupsTimeStateSize = args.mambaNGroups * args.mambaDState
        self.nGroups = args.mambaNGroups
        self.headDim = args.mambaDHead
        self.chunkSize = args.mambaChunkSize
        self.timeStepLimit = (0.0, Float.infinity)
        self.timeStepMin = 0.001
        self.timeStepMax = 0.1

        self.convDim = intermediateSize + 2 * nGroups * ssmStateSize

        self.conv1d = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            padding: convKernelSize - 1,
            groups: convDim,
            bias: useConvBias
        )

        let projectionSize = intermediateSize + convDim + numHeads
        _inProj.wrappedValue = Linear(
            hiddenSize,
            projectionSize,
            bias: args.mambaProjBias
        )

        _dtBias.wrappedValue = MLXArray.ones([numHeads])

        let A = MLXArray(Array(1 ... numHeads)).asType(.float32)

        _aLog.wrappedValue = log(A)

        self.mambaRMSNorm = args.mambaRMSNorm
        if mambaRMSNorm {
            self.norm = RMSNormGated(
                hiddenSize: intermediateSize,
                eps: layerNormEpsilon,
                nGroups: nGroups,
                normBeforeGate: args.mambaNormBeforeGate
            )
        } else {
            self.norm = nil
        }

        _d.wrappedValue = MLXArray.ones([numHeads]) + 1.0

        _outProj.wrappedValue = Linear(
            intermediateSize,
            hiddenSize,
            bias: args.projectorsBias
        )

        self.ssmInMultiplier = args.ssmInMultiplier
        self._mupVector = mupVector
    }

    func callAsFunction(
        _ inputStates: MLXArray, cache: Mamba2Cache? = nil, mask: MLXArray? = nil,
        cachePosition: MLXArray? = nil
    ) -> MLXArray {
        let (batchSize, seqLen, _) = (inputStates.dim(0), inputStates.dim(1), inputStates.dim(2))
        let dtype = inputStates.dtype

        let mask: MLXArray? = mask?[..<1, .ellipsis]

        var inputStates = applyMaskToPaddingStates(inputStates, mask)

        inputStates = inputStates * ssmInMultiplier
        var projectedStates = inProj(inputStates)
        projectedStates = projectedStates * _mupVector

        let gate = projectedStates[.ellipsis, ..<intermediateSize]
        var hiddenStatesBC = projectedStates[
            .ellipsis, intermediateSize ..< (intermediateSize + convDim)]
        let dt = projectedStates[.ellipsis, (intermediateSize + convDim)...]

        let usePrecomputedStates: Bool =
            cache != nil && cache!.hasPreviousState && seqLen == 1
            && cache!.convStates[layerIdx]!.shape[0] == batchSize
            && cache!.ssmStates[layerIdx]!.shape[0] == batchSize && cachePosition != nil
            && cachePosition![0].all().item() > 0

        if usePrecomputedStates, let cache {
            var convState = roll(cache.convStates[layerIdx]!, shift: -1, axis: -1)
            convState[0..., 0..., -1] = hiddenStatesBC[0..., 0, 0...]
            cache.convStates[layerIdx] = convState

            hiddenStatesBC = sum(convState * squeezed(conv1d.weight, axis: -1), axis: -1)
            if useConvBias {
                hiddenStatesBC = hiddenStatesBC + conv1d.bias!
            }
            hiddenStatesBC = silu(hiddenStatesBC)
        } else {
            if let cache {
                let hiddenStatesBCTransposed = hiddenStatesBC.transposed(0, 2, 1)
                let seqLenTransposed: Int = hiddenStatesBCTransposed.dim(-1)
                let padSize = convKernelSize - seqLenTransposed

                let convStates: MLXArray =
                    if padSize > 0 {
                        padded(
                            hiddenStatesBCTransposed,
                            widths: [.init((0, 0)), .init((0, 0)), .init((padSize, 0))])
                    } else {
                        hiddenStatesBCTransposed[0..., 0..., ..<padSize]
                    }

                cache.convStates[layerIdx] = convStates
            }

            hiddenStatesBC = silu(conv1d(hiddenStatesBC))[0..., ..<seqLen, 0...]
        }

        hiddenStatesBC = applyMaskToPaddingStates(hiddenStatesBC, mask)

        var hiddenStates = hiddenStatesBC[.ellipsis, ..<intermediateSize]
        let B = hiddenStatesBC[
            .ellipsis, intermediateSize ..< (intermediateSize + nGroups * ssmStateSize)]
        let C = hiddenStatesBC[.ellipsis, (intermediateSize + nGroups * ssmStateSize)...]

        let A = -exp(aLog.asType(.float32))

        if usePrecomputedStates {
            var dt = dt[0..., 0, 0...][0..., .newAxis, 0...]
            dt = dt.transposed(0, 2, 1)
            dt = broadcast(dt, to: [batchSize, dt.dim(1), headDim])

            var dtBias = expandedDimensions(dtBias, axis: -1)
            dtBias = broadcast(dtBias, to: [dtBias.dim(0), headDim])

            dt = softplus(dt + dtBias.asType(dt.dtype))
            dt = clip(dt, min: timeStepLimit.0, max: timeStepLimit.1)

            var A = expandedDimensions(expandedDimensions(A, axis: -1), axis: -1)
            A = broadcast(A, to: [numHeads, headDim, ssmStateSize]).asType(.float32)

            let dA = exp(expandedDimensions(dt, axis: -1) * A)

            var B = B.reshaped(batchSize, nGroups, -1)
            B = expandedDimensions(B, axis: 2)
            B = broadcast(B, to: [batchSize, nGroups, Int(numHeads / nGroups), B.dim(-1)])
            B = B.reshaped(batchSize, -1, B.dim(-1))

            let dB = expandedDimensions(dt, axis: -1) * expandedDimensions(B, axis: 2)

            hiddenStates = hiddenStates.reshaped(batchSize, -1, headDim)
            let dBx = dB * expandedDimensions(hiddenStates, axis: -1)

            let newSsmState = cache!.ssmStates[layerIdx]! * dA + dBx
            cache!.ssmStates[layerIdx] = newSsmState

            var C = C.reshaped(batchSize, nGroups, -1)
            C = expandedDimensions(C, axis: 2)
            C = broadcast(C, to: [batchSize, nGroups, Int(numHeads / nGroups), C.dim(-1)])
            C = C.reshaped(batchSize, -1, C.dim(-1))

            let ssmStates = cache!.ssmStates[layerIdx]!.asType(C.dtype)

            let ssmStatesReshaped = ssmStates.reshaped(batchSize * numHeads, headDim, ssmStateSize)
            let CReshaped = C.reshaped(batchSize * numHeads, ssmStateSize, 1)

            var y = matmul(ssmStatesReshaped, CReshaped)
            y = y.reshaped(batchSize, numHeads, headDim)

            var D = expandedDimensions(d, axis: -1)
            D = broadcast(D, to: [d.dim(0), headDim])
            y = y + hiddenStates * D

            y = y.reshaped(batchSize, -1)
            y = expandedDimensions(y, axis: 1)

            let scanOutput: MLXArray =
                if let norm {
                    norm(y, gate: gate)
                } else {
                    y * silu(gate)
                }

            return outProj(scanOutput.asType(dtype))
        } else {
            var dt = softplus(dt + dtBias)
            dt = clip(dt, min: timeStepLimit.0, max: timeStepLimit.1)

            hiddenStates = hiddenStates.reshaped(batchSize, seqLen, -1, headDim).asType(.float32)
            var B = B.reshaped(batchSize, seqLen, -1, ssmStateSize).asType(.float32)
            var C = C.reshaped(batchSize, seqLen, -1, ssmStateSize).asType(.float32)

            B = repeated(B, count: numHeads / nGroups, axis: 2)
            C = repeated(C, count: numHeads / nGroups, axis: 2)

            let padSize = (chunkSize - seqLen % chunkSize) % chunkSize

            let DResidual = expandedDimensions(d, axis: -1) * padTensorBySize(hiddenStates, padSize)

            hiddenStates = hiddenStates * expandedDimensions(dt, axis: -1)
            var A = A.asType(hiddenStates.dtype) * dt

            hiddenStates = reshapeIntoChunks(hiddenStates, padSize, chunkSize)
            A = reshapeIntoChunks(A, padSize, chunkSize)
            B = reshapeIntoChunks(B, padSize, chunkSize)
            C = reshapeIntoChunks(C, padSize, chunkSize)

            A = A.transposed(0, 3, 1, 2)
            let ACumsum = cumsum(A, axis: -1)

            let L = exp(segmentSum(A))

            var CExpanded = expandedDimensions(C, axis: 3)
            let BExpanded = expandedDimensions(B, axis: 2)
            let GIntermediate = CExpanded * BExpanded
            let G = sum(GIntermediate, axis: -1)

            let LPermuted = L.transposed(0, 2, 3, 4, 1)
            let MIntermediate =
                expandedDimensions(G, axis: -1) * expandedDimensions(LPermuted, axis: -1)
            let M = sum(MIntermediate, axis: -1)

            var hiddenStatesExpanded = expandedDimensions(hiddenStates, axis: 2)
            let MExpanded = expandedDimensions(M, axis: -1)
            let YDiag = sum(MExpanded * hiddenStatesExpanded, axis: 3)

            let decayStates = exp(ACumsum[0..., 0..., 0..., (-1)...] - ACumsum)
            let decayStatesPermuted = decayStates.transposed(0, 2, 3, 1)
            let BDecay = B * expandedDimensions(decayStatesPermuted, axis: -1)

            let BDecayExpanded = expandedDimensions(BDecay, axis: -2)
            hiddenStatesExpanded = expandedDimensions(hiddenStates, axis: -1)
            var states = sum(BDecayExpanded * hiddenStatesExpanded, axis: 2)

            let previousStates: MLXArray =
                if usePrecomputedStates {
                    expandedDimensions(cache!.ssmStates[layerIdx]!, axis: 1)
                } else {
                    MLXArray.zeros(like: states[0..., ..<1])
                }

            states = concatenated([previousStates, states], axis: 1)

            let ACumsumLast = ACumsum[0..., 0..., 0..., -1]
            let padded = padded(ACumsumLast, widths: [.init((0, 0)), .init((0, 0)), .init((1, 0))])
            var decayChunk = exp(segmentSum(padded))
            decayChunk = decayChunk.transposed(0, 3, 2, 1)

            let decayExpanded = expandedDimensions(
                expandedDimensions(decayChunk, axis: -1), axis: -1)
            var statesExpanded = expandedDimensions(states, axis: 2)
            let newStates = sum(decayExpanded * statesExpanded, axis: 1)

            states = newStates[0..., ..<(-1)]
            let ssmState = newStates[0..., -1]

            let stateDecayOut = exp(ACumsum)
            CExpanded = expandedDimensions(C, axis: -2)
            statesExpanded = expandedDimensions(states, axis: 2)
            let CTimesStates = CExpanded * statesExpanded

            let stateDecayOutPermuted = stateDecayOut.transposed(0, 2, 3, 1)
            let CTimesStatesSum = sum(CTimesStates, axis: -1)
            let YOff = CTimesStatesSum * expandedDimensions(stateDecayOutPermuted, axis: -1)

            var y = YDiag + YOff

            y = y.reshaped(batchSize, -1, numHeads, headDim)
            y = y + DResidual

            if padSize > 0 {
                y = y[0..., ..<seqLen, 0..., 0...]
            }
            y = y.reshaped(batchSize, seqLen, -1)

            if let cache {
                cache.ssmStates[layerIdx] = ssmState
                cache.hasPreviousState = true
            }

            let scanOutput: MLXArray =
                if let norm {
                    norm(y, gate: gate)
                } else {
                    y * silu(gate)
                }

            return outProj(scanOutput.asType(dtype))
        }
    }
}

// MARK: - Model

private class ModelInner: Module {
    let args: FalconH1Configuration
    let vocabSize: Int
    let hiddenSize: Int

    fileprivate let layers: [DecoderLayer]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: RMSNorm

    init(_ args: FalconH1Configuration) {
        self.args = args
        self.vocabSize = args.vocabSize
        self.hiddenSize = args.hiddenSize

        precondition(vocabSize > 0)

        _embedTokens.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: hiddenSize)

        let mupVector = computeMupVector(args)
        self.layers = (0 ..< args.numHiddenLayers).map { layerIdx in
            DecoderLayer(args, layerIdx: layerIdx, mupVector: mupVector)
        }

        _finalLayerNorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil, cache: [Mamba2Cache]? = nil)
        -> MLXArray
    {
        var h = embedTokens(inputs)

        h = h * args.embeddingMultiplier

        let mask = mask ?? createAttentionMask(h: h, cache: nil)
        let mambaMask: MLXArray? = nil

        let cachePosition = MLXArray(0 ..< h.dim(1)).asType(.int32)

        if h.dim(1) == 1, let cache {
            let prevSeqlen = cache[0].keyCache[0].dim(-2)
            let cachePosition = cachePosition + prevSeqlen
        }

        for (layer, c) in zip(layers, cache!) {
            h = layer(
                h,
                cache: c ?? Mamba2Cache(args),
                mask: nil,
                mambaMask: mambaMask,
                cachePosition: cachePosition
            )
        }

        return finalLayerNorm(h)
    }
}

public class FalconH1Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: ModelInner
    let configuration: FalconH1Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: FalconH1Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabSize
        self.kvHeads = (0 ..< args.numKeyValueHeads).map { _ in args.numHiddenLayers }
        self.model = ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache as? [Mamba2Cache])
        if let lmHead {
            out = lmHead(out) * configuration.lmHeadMultiplier
        } else {
            out = model.embedTokens.asLinear(out)
        }

        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        for (name, param) in weights {
            if name.contains("conv1d.weight"), param.dim(-1) > param.dim(1) {
                weights[name] = param.transposed(0, 2, 1)
            }
        }

        return weights
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        model.layers.map { _ in Mamba2Cache(configuration) }
    }
}

// MARK: - LoRA

extension FalconH1Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}

// MARK: - Configuration

public struct FalconH1Configuration: Codable, Sendable {
    var attentionBias: Bool
    var attentionDropout: Float
    var attentionInMultiplier: Float
    var attentionOutMultiplier: Float
    var bosTokenId: Int
    var embeddingMultiplier: Float
    var eosTokenId: Int
    var headDim: Int?
    var hiddenAct: String
    var hiddenSize: Int
    var initializerRange: Float
    var intermediateSize: Int
    var keyMultiplier: Float
    var lmHeadMultiplier: Float
    var mambaChunkSize: Int
    var mambaConvBias: Bool
    var mambaDConv: Int
    var mambaDHead: Int
    var mambaDSSM: Int?
    var mambaDState: Int
    var mambaExpand: Int
    var mambaNGroups: Int
    var mambaNHeads: Int
    var mambaNormBeforeGate: Bool
    var mambaProjBias: Bool
    var mambaRMSNorm: Bool
    var mambaUseMLP: Bool
    var maxPositionEmbeddings: Int
    var mlpBias: Bool
    var mlpExpansionFactor: Int
    var mlpMultipliers: [Float]
    var modelType: String
    var numAttentionHeads: Int
    var numHiddenLayers: Int
    var numKeyValueHeads: Int
    var numLogitsToKeep: Int
    var padTokenId: Int
    var projectorsBias: Bool
    var rmsNormEps: Float
    var ropeTraditional: Bool
    var ropeScaling: Float?
    var ropeTheta: Float
    var ssmInMultiplier: Float
    var ssmMultipliers: [Float]
    var ssmOutMultiplier: Float
    var tieWordEmbeddings: Bool
    var torchDtype: String
    var vocabSize: Int

    enum CodingKeys: String, CodingKey {
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case attentionInMultiplier = "attention_in_multiplier"
        case attentionOutMultiplier = "attention_out_multiplier"
        case bosTokenId = "bos_token_id"
        case embeddingMultiplier = "embedding_multiplier"
        case eosTokenId = "eos_token_id"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case hiddenSize = "hidden_size"
        case initializerRange = "initializer_range"
        case intermediateSize = "intermediate_size"
        case keyMultiplier = "key_multiplier"
        case lmHeadMultiplier = "lm_head_multiplier"
        case mambaChunkSize = "mamba_chunk_size"
        case mambaConvBias = "mamba_conv_bias"
        case mambaDConv = "mamba_d_conv"
        case mambaDHead = "mamba_d_head"
        case mambaDSSM = "mamba_d_ssm"
        case mambaDState = "mamba_d_state"
        case mambaExpand = "mamba_expand"
        case mambaNGroups = "mamba_n_groups"
        case mambaNHeads = "mamba_n_heads"
        case mambaNormBeforeGate = "mamba_norm_before_gate"
        case mambaProjBias = "mamba_proj_bias"
        case mambaRMSNorm = "mamba_rms_norm"
        case mambaUseMLP = "mamba_use_mlp"
        case maxPositionEmbeddings = "max_position_embeddings"
        case mlpBias = "mlp_bias"
        case mlpExpansionFactor = "mlp_expansion_factor"
        case mlpMultipliers = "mlp_multipliers"
        case modelType = "model_type"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case numLogitsToKeep = "num_logits_to_keep"
        case padTokenId = "pad_token_id"
        case projectorsBias = "projectors_bias"
        case rmsNormEps = "rms_norm_eps"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case ropeTheta = "rope_theta"
        case ssmInMultiplier = "ssm_in_multiplier"
        case ssmMultipliers = "ssm_multipliers"
        case ssmOutMultiplier = "ssm_out_multiplier"
        case tieWordEmbeddings = "tie_word_embeddings"
        case torchDtype = "torch_dtype"
        case vocabSize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.attentionDropout =
            try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        self.attentionInMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .attentionInMultiplier) ?? 1.0
        self.attentionOutMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .attentionOutMultiplier) ?? 1.0
        self.bosTokenId = try container.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 1
        self.embeddingMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .embeddingMultiplier) ?? 1.0
        self.eosTokenId = try container.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 2
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? nil
        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.initializerRange =
            try container.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.keyMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .keyMultiplier) ?? 1.0
        self.lmHeadMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .lmHeadMultiplier) ?? 1.0
        self.mambaChunkSize =
            try container.decodeIfPresent(Int.self, forKey: .mambaChunkSize) ?? 256
        self.mambaConvBias =
            try container.decodeIfPresent(Bool.self, forKey: .mambaConvBias) ?? true
        self.mambaDConv = try container.decodeIfPresent(Int.self, forKey: .mambaDConv) ?? 4
        self.mambaDHead = try container.decodeIfPresent(Int.self, forKey: .mambaDHead) ?? 64
        self.mambaDSSM = try container.decodeIfPresent(Int.self, forKey: .mambaDSSM) ?? nil
        self.mambaDState = try container.decodeIfPresent(Int.self, forKey: .mambaDState) ?? 256
        self.mambaExpand = try container.decodeIfPresent(Int.self, forKey: .mambaExpand) ?? 2
        self.mambaNGroups = try container.decodeIfPresent(Int.self, forKey: .mambaNGroups) ?? 1
        self.mambaNHeads = try container.decodeIfPresent(Int.self, forKey: .mambaNHeads) ?? 128
        self.mambaNormBeforeGate =
            try container.decodeIfPresent(Bool.self, forKey: .mambaNormBeforeGate) ?? true
        self.mambaProjBias =
            try container.decodeIfPresent(Bool.self, forKey: .mambaProjBias) ?? false
        self.mambaRMSNorm = try container.decodeIfPresent(Bool.self, forKey: .mambaRMSNorm) ?? false
        self.mambaUseMLP = try container.decodeIfPresent(Bool.self, forKey: .mambaUseMLP) ?? true
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        self.mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        self.mlpExpansionFactor =
            try container.decodeIfPresent(Int.self, forKey: .mlpExpansionFactor) ?? 8
        self.mlpMultipliers =
            try container.decodeIfPresent([Float].self, forKey: .mlpMultipliers) ?? [1.0, 1.0]
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "falcon_h1"
        self.numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        self.numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 32
        self.numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        self.numLogitsToKeep =
            try container.decodeIfPresent(Int.self, forKey: .numLogitsToKeep) ?? 1
        self.padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 0
        self.projectorsBias =
            try container.decodeIfPresent(Bool.self, forKey: .projectorsBias) ?? false
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-05
        self.ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.ropeScaling = try container.decodeIfPresent(Float?.self, forKey: .ropeScaling) ?? nil
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
        self.ssmInMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .ssmInMultiplier) ?? 1.0
        self.ssmMultipliers =
            try container.decodeIfPresent([Float].self, forKey: .ssmMultipliers) ?? [
                1.0, 1.0, 1.0, 1.0, 1.0,
            ]
        self.ssmOutMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .ssmOutMultiplier) ?? 1.0
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.torchDtype =
            try container.decodeIfPresent(String.self, forKey: .torchDtype) ?? "bfloat16"
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 128000
    }
}

// MARK: - Mamba2Cache KVCache

private class Mamba2Cache: KVCache {
    var offset: Int

    var maxSize: Int?

    func innerState() -> [MLXArray] {
        []
    }

    var seqlenOffset: Int = 0
    var hasPreviousState: Bool = false
    let convKernelSize: Int

    private var _seenTokens: Int = 0

    let intermediateSize: Int

    var convStates: [Int: MLXArray]
    var ssmStates: [Int: MLXArray]

    var transformerLayers: [Int]
    var keyCache: [MLXArray]
    var valueCache: [MLXArray]

    init(_ args: FalconH1Configuration, batchSize: Int = 1) {
        self.convKernelSize = args.mambaDConv

        self.intermediateSize =
            args.mambaDSSM ?? Int(Float(args.mambaExpand) * Float(args.hiddenSize))

        self.convStates = [:]
        self.ssmStates = [:]

        for i in 0 ..< args.numHiddenLayers {
            convStates[i] = MLXArray.zeros([
                batchSize,
                intermediateSize + 2 * args.mambaNGroups * args.mambaDState,
                convKernelSize,
            ])
            ssmStates[i] = MLXArray.zeros([
                batchSize,
                args.mambaNHeads,
                args.mambaDHead,
                args.mambaDState,
            ])
        }

        self.seqlenOffset = 0
        self.hasPreviousState = false
        self.transformerLayers = Array(0 ..< args.numHiddenLayers)
        self.keyCache = []
        self.valueCache = []
        self.offset = 0
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        update(keyStates: keys, valueStates: values, layerIdx: 0)
    }

    func update(keyStates: MLXArray, valueStates: MLXArray, layerIdx: Int) -> (MLXArray, MLXArray) {
        if layerIdx == 0 {
            _seenTokens += keyStates.dim(-2)
        }

        if keyCache.count <= layerIdx {
            for _ in keyCache.count ..< layerIdx {
                keyCache.append(MLXArray([]))
                valueCache.append(MLXArray([]))
            }
            keyCache.append(keyStates)
            valueCache.append(valueStates)
        } else if keyCache[layerIdx].size == 0 {
            keyCache[layerIdx] = keyStates
            valueCache[layerIdx] = valueStates
        } else {
            keyCache[layerIdx] = concatenated([keyCache[layerIdx], keyStates], axis: -2)
            valueCache[layerIdx] = concatenated([valueCache[layerIdx], valueStates], axis: -2)
        }

        return (keyCache[layerIdx], valueCache[layerIdx])
    }

    func updateConvState(layerIdx: Int, newConvState: MLXArray, cachePosition: MLXArray) -> MLXArray
    {
        var convState = convStates[layerIdx]!
        let cachePosition = clip(cachePosition, min: 0, max: convKernelSize - 1)

        convState = roll(convState, shift: -1, axis: -1)

        if cachePosition.count > 1 {
            convState[0..., 0..., 0...] = newConvState.transposed(0, 2, 1)
        } else {
            convState[0..., 0..., -1] = newConvState[0..., 0..., -1]
        }

        convStates[layerIdx] = convState
        return convStates[layerIdx]!
    }

    func reset() {
        for i in 0 ..< convStates.count {
            convStates[i] = MLXArray.zeros(like: convStates[i]!)
            ssmStates[i] = MLXArray.zeros(like: ssmStates[i]!)
        }
    }
}
