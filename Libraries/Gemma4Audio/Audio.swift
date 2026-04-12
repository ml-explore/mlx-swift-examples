// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXFast

/// RMSNorm with weight applied directly (no offset).
open class AudioRMSNorm: Module, UnaryLayer {
    public let weight: MLXArray
    public let eps: Float

    public init(dim: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dim])
        self.eps = eps
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// Conv2d -> LayerNorm(channels) -> ReLU with symmetric padding.
open class SSCPConvBlock: Module {
    public let timeStride: Int = 2
    public let padding: (Int, Int, Int, Int) = (1, 1, 1, 1)  // top, bottom, left, right

    public let conv: Conv2d
    public let norm: LayerNorm

    public init(inChannels: Int, outChannels: Int, rmsNormEps: Float = 1e-6) {
        self.conv = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: [3, 3],
            stride: [2, 2],
            padding: [0, 0],  // manual padding applied in call
            bias: false
        )
        self.norm = LayerNorm(dimensions: outChannels, eps: rmsNormEps, affine: false)
        super.init()
    }

    /// Forward pass
    /// - Parameters:
    ///   - x: Input array of shape `[B, T, F, C]`
    ///   - mask: Mask array of shape `[B, T]` where True indicates invalid/padding
    /// - Returns: Tuple of output array and downsampled mask
    open func callAsFunction(_ x: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        // Zero out invalid positions
        var out = MLX.where(mask.expandedDimensions(axes: [2, 3]), MLXArray(0.0), x)

        // Manual padding on T and F dims
        // x shape is [B, T, F, C]. Pad T (axis 1) and F (axis 2)
        out = padded(
            out,
            widths: [
                0,  // B
                [padding.0, padding.1],  // T
                [padding.2, padding.3],  // F
                0,  // C
            ])

        out = conv(out)

        // Downsample mask by time stride
        let tOut = out.dim(1)
        // Equivalent to python: mask[:, :: self.time_stride][:, :t_out]
        let outputMask = mask[MLXEllipsisIndex.ellipsis, stride(by: timeStride)][
            MLXEllipsisIndex.ellipsis, 0 ..< tOut]

        // LayerNorm over channels (last dim)
        out = norm(out)
        out = MLXNN.relu(out)

        return (out, outputMask)
    }
}

/// SSCP: 2 Conv2d blocks -> flatten(F, C) -> Linear projection to hidden_size.
open class SubSampleConvProjection: Module {
    public static let inputFeatSize = 128

    public let layer0: SSCPConvBlock
    public let layer1: SSCPConvBlock
    public let inputProjLinear: Linear

    public init(
        hiddenSize: Int, subsamplingConvChannels: [Int] = [256, 256], rmsNormEps: Float = 1e-6
    ) {
        precondition(
            subsamplingConvChannels.count >= 2,
            "subsamplingConvChannels must have at least 2 elements")

        self.layer0 = SSCPConvBlock(
            inChannels: 1, outChannels: subsamplingConvChannels[0], rmsNormEps: rmsNormEps)
        self.layer1 = SSCPConvBlock(
            inChannels: subsamplingConvChannels[0], outChannels: subsamplingConvChannels[1],
            rmsNormEps: rmsNormEps)

        var freq = SubSampleConvProjection.inputFeatSize
        for _ in 0 ..< 2 {
            freq = (freq + 2 - 3) / 2 + 1
        }
        let projInputDim = freq * subsamplingConvChannels[1]
        self.inputProjLinear = Linear(projInputDim, hiddenSize, bias: false)
        super.init()
    }

    open func callAsFunction(_ audioMel: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        // audio_mel: [B, T, F_in]
        // Add channel dim: [B, T, F, 1]
        var x = audioMel.expandedDimensions(axes: [-1])
        var currentMask = mask

        let res0 = layer0(x, mask: currentMask)
        x = res0.0
        currentMask = res0.1

        let res1 = layer1(x, mask: currentMask)
        x = res1.0
        currentMask = res1.1

        // Flatten F*C -> [B, T, F*C]
        let b = x.dim(0)
        let t = x.dim(1)
        let f = x.dim(2)
        let c = x.dim(3)
        x = x.reshaped([b, t, f * c])

        // Project to hidden_size
        x = inputProjLinear(x)

        return (x, currentMask)
    }
}

/// Configuration for the audio model
public struct AudioConfig {
    public let hiddenSize: Int
    public let numAttentionHeads: Int
    public let convKernelSize: Int
    public let subsamplingConvChannels: [Int]
    public let rmsNormEps: Float
    public let gradientClipping: Float
    public let residualWeight: Float
    public let attentionContextLeft: Int
    public let attentionContextRight: Int
    public let attentionChunkSize: Int
    public let attentionInvalidLogitsValue: Float
    public let attentionLogitCap: Float

    public init(
        hiddenSize: Int = 1024,
        numAttentionHeads: Int = 8,
        convKernelSize: Int = 31,
        subsamplingConvChannels: [Int] = [256, 256],
        rmsNormEps: Float = 1e-6,
        gradientClipping: Float = 10.0,
        residualWeight: Float = 1.0,
        attentionContextLeft: Int = 32,
        attentionContextRight: Int = 0,
        attentionChunkSize: Int = 32,
        attentionInvalidLogitsValue: Float = -1e4,
        attentionLogitCap: Float = 50.0
    ) {
        self.hiddenSize = hiddenSize
        self.numAttentionHeads = numAttentionHeads
        self.convKernelSize = convKernelSize
        self.subsamplingConvChannels = subsamplingConvChannels
        self.rmsNormEps = rmsNormEps
        self.gradientClipping = gradientClipping
        self.residualWeight = residualWeight
        self.attentionContextLeft = attentionContextLeft
        self.attentionContextRight = attentionContextRight
        self.attentionChunkSize = attentionChunkSize
        self.attentionInvalidLogitsValue = attentionInvalidLogitsValue
        self.attentionLogitCap = attentionLogitCap
    }
}

/// Linear layer with optional input/output clamping.
open class ClippableLinear: Module, UnaryLayer {
    public let linear: Linear
    public let useClipping: Bool

    public var inputMin: MLXArray?
    public var inputMax: MLXArray?
    public var outputMin: MLXArray?
    public var outputMax: MLXArray?

    public init(_ inFeatures: Int, _ outFeatures: Int, bias: Bool = false, useClipping: Bool = true)
    {
        self.linear = Linear(inFeatures, outFeatures, bias: bias)
        self.useClipping = useClipping
        if useClipping {
            self.inputMin = MLXArray(-Float.infinity)
            self.inputMax = MLXArray(Float.infinity)
            self.outputMin = MLXArray(-Float.infinity)
            self.outputMax = MLXArray(Float.infinity)
        }
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        if useClipping, let imin = inputMin, let imax = inputMax {
            out = MLX.clip(out, min: imin, max: imax)
        }
        out = linear(out)
        if useClipping, let omin = outputMin, let omax = outputMax {
            out = MLX.clip(out, min: omin, max: omax)
        }
        return out
    }
}

/// Macaron-style FFW with residual scaling.
open class ConformerFeedForward: Module, UnaryLayer {
    public let gradientClipping: Float
    public let residualWeight: Float

    public let preLayerNorm: AudioRMSNorm
    public let ffwLayer1: ClippableLinear
    public let ffwLayer2: ClippableLinear
    public let postLayerNorm: AudioRMSNorm

    public init(config: AudioConfig) {
        self.gradientClipping = config.gradientClipping
        self.residualWeight = config.residualWeight

        self.preLayerNorm = AudioRMSNorm(dim: config.hiddenSize, eps: config.rmsNormEps)
        self.ffwLayer1 = ClippableLinear(config.hiddenSize, config.hiddenSize * 4, bias: false)
        self.ffwLayer2 = ClippableLinear(config.hiddenSize * 4, config.hiddenSize, bias: false)
        self.postLayerNorm = AudioRMSNorm(dim: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var out = MLX.clip(x, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        out = preLayerNorm(out)
        out = ffwLayer1(out)
        out = MLXNN.silu(out)
        out = ffwLayer2(out)
        out = MLX.clip(out, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        out = postLayerNorm(out)
        return residual + out * residualWeight
    }
}

/// Light convolution: norm -> linear(2x) -> GLU -> depthwise_conv1d(causal) -> norm -> SiLU -> linear + residual.
open class ConformerLightConv1d: Module, UnaryLayer {
    public let gradientClipping: Float
    public let causalPadding: Int

    public let preLayerNorm: AudioRMSNorm
    public let linearStart: ClippableLinear
    public let depthwiseConv1d: Conv1d
    public let convNorm: AudioRMSNorm
    public let linearEnd: ClippableLinear

    public init(config: AudioConfig) {
        self.gradientClipping = config.gradientClipping
        self.causalPadding = config.convKernelSize - 1

        self.preLayerNorm = AudioRMSNorm(dim: config.hiddenSize, eps: config.rmsNormEps)
        self.linearStart = ClippableLinear(config.hiddenSize, config.hiddenSize * 2, bias: false)

        self.depthwiseConv1d = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.convKernelSize,
            stride: 1,
            padding: 0,
            groups: config.hiddenSize,
            bias: false
        )

        self.convNorm = AudioRMSNorm(dim: config.hiddenSize, eps: config.rmsNormEps)
        self.linearEnd = ClippableLinear(config.hiddenSize, config.hiddenSize, bias: false)
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x

        var out = preLayerNorm(x)
        out = linearStart(out)

        // GLU: split in half along last dim and gate
        let split = MLX.split(out, parts: 2, axis: -1)
        out = split[0] * MLXNN.sigmoid(split[1])

        // Causal padding for Conv1d
        out = padded(
            out,
            widths: [
                0,  // B
                [causalPadding, 0],  // T (causal pad left)
                0,  // C
            ])

        out = depthwiseConv1d(out)

        out = MLX.clip(out, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        out = convNorm(out)
        out = MLXNN.silu(out)
        out = linearEnd(out)

        return out + residual
    }
}
