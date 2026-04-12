// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXFast

/// Sinusoidal relative position embedding for chunked attention.
open class AudioRelativePositionEmbedding: Module {
    public let numHeads: Int
    public let channels: Int
    public let headDim: Int
    public let maxBackward: Int
    public let maxForward: Int

    public let posProj: Linear
    public let invTimescales: MLXArray

    public init(config: AudioConfig) {
        self.numHeads = config.numAttentionHeads
        self.channels = config.hiddenSize
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.maxBackward = max(0, config.attentionContextLeft - 1)
        self.maxForward = config.attentionContextRight

        self.posProj = Linear(self.channels, self.numHeads * self.headDim, bias: false)

        let minTimescale: Float = 1.0
        let maxTimescale: Float = 10000.0
        let numTimescales = self.channels / 2
        let logTimescaleIncrement =
            log(maxTimescale / minTimescale) / Float(max(numTimescales - 1, 1))

        let arange = MLXArray(0 ..< Int32(numTimescales)).asType(.float32)
        let timescales = minTimescale * MLX.exp(arange * -logTimescaleIncrement)
        self.invTimescales = timescales.reshaped([1, 1, numTimescales])
        super.init()
    }

    // Overloaded init to be used internally by AudioAttention
    public init(
        numHeads: Int, channels: Int, headDim: Int, maxBackward: Int, maxForward: Int,
        posProj: Linear
    ) {
        self.numHeads = numHeads
        self.channels = channels
        self.headDim = headDim
        self.maxBackward = maxBackward
        self.maxForward = maxForward
        self.posProj = posProj

        let minTimescale: Float = 1.0
        let maxTimescale: Float = 10000.0
        let numTimescales = self.channels / 2
        let logTimescaleIncrement =
            log(maxTimescale / minTimescale) / Float(max(numTimescales - 1, 1))

        let arange = MLXArray(0 ..< Int32(numTimescales)).asType(.float32)
        let timescales = minTimescale * MLX.exp(arange * -logTimescaleIncrement)
        self.invTimescales = timescales.reshaped([1, 1, numTimescales])
        super.init()
    }

    public func getTimingSignal(_ position: MLXArray, dtype: DType) -> MLXArray {
        let pos = position.asType(.float32).expandedDimensions(axes: [-1])
        let scaledTime = pos * invTimescales
        let signal = MLX.concatenated([MLX.sin(scaledTime), MLX.cos(scaledTime)], axis: -1)
        return signal.asType(dtype)
    }

    public func relativeShift(
        termBd: MLXArray, batchSize: Int, numHeads: Int, numBlocks: Int, blockSize: Int,
        contextSize: Int, maxSpanPlus1: Int
    ) -> MLXArray {
        let padAmount = (contextSize + 1) - maxSpanPlus1
        var out = padded(
            termBd,
            widths: [
                0,
                0,
                0,
                0,
                [0, padAmount],
            ])
        out = out.reshaped([batchSize, numHeads, numBlocks, blockSize * (contextSize + 1)])
        out = out[MLXEllipsisIndex.ellipsis, 0 ..< (blockSize * contextSize)]
        out = out.reshaped([batchSize, numHeads, numBlocks, blockSize, contextSize])
        return out
    }

    open func callAsFunction(queries: MLXArray, keys: MLXArray) -> MLXArray {
        let b = queries.dim(0)
        let u = queries.dim(1)
        let w = queries.dim(2)
        let n = queries.dim(3)
        let h = queries.dim(4)
        let c = keys.dim(2)

        // pos_indices = mx.arange(self.max_backward, -self.max_forward - 1, -1)[None]
        let posIndices = MLXArray(Array(stride(from: maxBackward, to: -maxForward - 1, by: -1)))
            .expandedDimensions(axes: [0])
        let maxSpanPlus1 = posIndices.dim(1)

        var sinEmb = getTimingSignal(posIndices, dtype: queries.dtype)
        sinEmb = posProj(sinEmb.asType(posProj.weight.dtype))
        sinEmb = sinEmb.reshaped([maxSpanPlus1, numHeads, headDim])
        sinEmb = sinEmb.asType(queries.dtype)

        let queriesP = queries.transposed(axes: [0, 3, 1, 2, 4])
        let keysP = keys.transposed(axes: [0, 3, 1, 4, 2])
        let termAc = MLX.matmul(queriesP, keysP)

        let sinEmbT = sinEmb.transposed(axes: [1, 2, 0])
        let qReshaped = queriesP.reshaped([b, n, u * w, h])
        var termBd = MLX.matmul(qReshaped, sinEmbT)
        termBd = termBd.reshaped([b, n, u, w, maxSpanPlus1])

        termBd = relativeShift(
            termBd: termBd, batchSize: b, numHeads: n, numBlocks: u, blockSize: w, contextSize: c,
            maxSpanPlus1: maxSpanPlus1)

        return termAc + termBd
    }
}

/// Chunked local attention with relative position embeddings and logit softcapping.
open class AudioAttention: Module {
    public let numHeads: Int
    public let hiddenSize: Int
    public let headDim: Int

    public let chunkSize: Int
    public let maxFutureHorizon: Int
    public let maxPastHorizon: Int
    public let contextSize: Int
    public let invalidLogitsValue: Float
    public let softcap: Float

    public let relativeKProj: Linear
    public var perDimScale: MLXArray

    public let qProj: ClippableLinear
    public let kProj: ClippableLinear
    public let vProj: ClippableLinear
    public let post: ClippableLinear

    public let qScale: Float
    public let kScale: Float

    public let relPos: AudioRelativePositionEmbedding

    public init(config: AudioConfig) {
        self.numHeads = config.numAttentionHeads
        self.hiddenSize = config.hiddenSize
        self.headDim = config.hiddenSize / config.numAttentionHeads

        self.chunkSize = config.attentionChunkSize
        self.maxFutureHorizon = config.attentionContextRight
        self.maxPastHorizon = max(0, config.attentionContextLeft - 1)
        self.contextSize = self.chunkSize + self.maxPastHorizon + self.maxFutureHorizon
        self.invalidLogitsValue = config.attentionInvalidLogitsValue
        self.softcap = config.attentionLogitCap

        self.relativeKProj = Linear(self.hiddenSize, self.numHeads * self.headDim, bias: false)
        self.perDimScale = MLXArray.zeros([self.headDim])

        self.qProj = ClippableLinear(self.hiddenSize, self.numHeads * self.headDim, bias: false)
        self.kProj = ClippableLinear(self.hiddenSize, self.numHeads * self.headDim, bias: false)
        self.vProj = ClippableLinear(self.hiddenSize, self.numHeads * self.headDim, bias: false)
        self.post = ClippableLinear(self.hiddenSize, self.hiddenSize, bias: false)

        self.qScale = Float(pow(Double(self.headDim), -0.5)) / Float(log(2.0))
        self.kScale = Float(log(1.0 + M_E)) / Float(log(2.0))

        self.relPos = AudioRelativePositionEmbedding(
            numHeads: self.numHeads,
            channels: self.hiddenSize,
            headDim: self.headDim,
            maxBackward: self.maxPastHorizon,
            maxForward: self.maxFutureHorizon,
            posProj: self.relativeKProj
        )
        super.init()
    }

    public func padDim1(_ x: MLXArray, padLeft: Int, padRight: Int) -> MLXArray {
        var widths = [IntOrPair](repeating: 0, count: x.ndim)
        widths[1] = [padLeft, padRight]
        return padded(x, widths: widths)
    }

    public func convertToBlock(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let rest = Array(x.shape.suffix(from: 2))

        let numBlocks = (t + chunkSize - 1) / chunkSize
        let padLen = numBlocks * chunkSize - t

        var out = x
        if padLen > 0 {
            out = padDim1(out, padLeft: 0, padRight: padLen)
        }

        var newShape = [b, numBlocks, chunkSize]
        newShape.append(contentsOf: rest)
        return out.reshaped(newShape)
    }

    public func extractBlockContext(_ x: MLXArray) -> MLXArray {
        let padLeft = maxPastHorizon
        let padRight = maxFutureHorizon + chunkSize - 1
        let out = padDim1(x, padLeft: padLeft, padRight: padRight)

        let tPadded = out.dim(1)
        let numBlocks = (tPadded - contextSize) / chunkSize + 1

        let starts = MLXArray(stride(from: 0, to: numBlocks, by: 1)) * chunkSize
        let offsets = MLXArray(stride(from: 0, to: contextSize, by: 1))
        let indices = starts.expandedDimensions(axes: [1]) + offsets.expandedDimensions(axes: [0])

        return out[MLXEllipsisIndex.ellipsis, indices]
    }

    open func callAsFunction(hiddenStates: MLXArray, mask: MLXArray, causalValidMask: MLXArray)
        -> MLXArray
    {
        let b = hiddenStates.dim(0)
        let t = hiddenStates.dim(1)
        let qkvShape = [b, t, numHeads, headDim]

        var q = qProj(hiddenStates).asType(.float32).reshaped(qkvShape)
        var k = kProj(hiddenStates).asType(.float32).reshaped(qkvShape)
        var v = vProj(hiddenStates).asType(.float32).reshaped(qkvShape)

        let perDimScaleNorm = MLXNN.softplus(perDimScale)
        q = q * (qScale * perDimScaleNorm)
        k = k * kScale

        let queryBlocks = convertToBlock(q)
        let keyBlocks = extractBlockContext(k)
        let valueBlocks = extractBlockContext(v)
        let u = queryBlocks.dim(1)

        let validMask = MLX.logicalNot(mask)
        let extractedValid = extractBlockContext(validMask)
        let condition = MLX.logicalAnd(
            extractedValid.expandedDimensions(axes: [1, 3]),
            causalValidMask.expandedDimensions(axes: [0, 1, 2])
        )

        var logits = relPos(queries: queryBlocks, keys: keyBlocks)
        logits = MLX.tanh(logits / softcap) * softcap
        logits = MLX.where(condition, logits, MLXArray(invalidLogitsValue))

        let probs = MLX.softmax(logits, axis: -1)
        var context = MLX.einsum("bnuwc,bucnh->buwnh", probs, valueBlocks)
        context = context.reshaped([b, u * chunkSize, numHeads, headDim])
        context = context[MLXEllipsisIndex.ellipsis, 0 ..< t]

        let bOut = context.dim(0)
        let tOut = context.dim(1)
        context = context.reshaped([bOut, tOut, numHeads * headDim])

        return post(context)
    }
}

/// Macaron-style Conformer block.
open class ConformerBlock: Module {
    public let gradientClipping: Float
    public let feedForward1: ConformerFeedForward
    public let selfAttn: AudioAttention
    public let lconv1d: ConformerLightConv1d
    public let feedForward2: ConformerFeedForward
    public let normPreAttn: AudioRMSNorm
    public let normPostAttn: AudioRMSNorm
    public let normOut: AudioRMSNorm

    public init(config: AudioConfig) {
        self.gradientClipping = config.gradientClipping
        self.feedForward1 = ConformerFeedForward(config: config)
        self.selfAttn = AudioAttention(config: config)
        self.lconv1d = ConformerLightConv1d(config: config)
        self.feedForward2 = ConformerFeedForward(config: config)
        self.normPreAttn = AudioRMSNorm(dim: config.hiddenSize)
        self.normPostAttn = AudioRMSNorm(dim: config.hiddenSize)
        self.normOut = AudioRMSNorm(dim: config.hiddenSize)
        super.init()
    }

    open func callAsFunction(_ x: MLXArray, mask: MLXArray, causalValidMask: MLXArray) -> MLXArray {
        var out = feedForward1(x)

        // Attention with pre/post norm and residual
        var residual = out
        out = MLX.clip(out, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        out = normPreAttn(out)
        out = selfAttn(hiddenStates: out, mask: mask, causalValidMask: causalValidMask)
        out = MLX.clip(out, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        out = residual + normPostAttn(out)

        // Zero out invalid positions before lconv1d
        let validityMask = MLX.logicalNot(mask).expandedDimensions(axes: [2]).asType(out.dtype)
        out = out * validityMask

        out = lconv1d(out)
        out = feedForward2(out)
        out = MLX.clip(out, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))

        return normOut(out)
    }
}

open class AudioEncoder: Module {
    public let pre: SubSampleConvProjection
    public let layers: [ConformerBlock]
    public let post: AudioRMSNorm

    public init(config: AudioConfig, numHiddenLayers: Int = 12) {
        self.pre = SubSampleConvProjection(
            hiddenSize: config.hiddenSize, subsamplingConvChannels: config.subsamplingConvChannels,
            rmsNormEps: config.rmsNormEps)
        self.layers = (0 ..< numHiddenLayers).map { _ in ConformerBlock(config: config) }
        self.post = AudioRMSNorm(dim: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    open func callAsFunction(_ audioMel: MLXArray, mask: MLXArray, causalValidMask: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        var (x, currentMask) = pre(audioMel, mask: mask)

        for layer in layers {
            x = layer(x, mask: currentMask, causalValidMask: causalValidMask)
        }

        x = post(x)
        return (x, currentMask)
    }
}

open class MultimodalEmbedder: Module {
    public let embeddingProjection: Linear

    public init(config: AudioConfig, textConfigHiddenSize: Int) {
        self.embeddingProjection = Linear(config.hiddenSize, textConfigHiddenSize, bias: false)
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        return embeddingProjection(x)
    }
}
