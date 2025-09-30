//
//  RoPEUtils.swift
//  mlx-swift-examples
//
//  Created by John Mai on 2025/8/11.
//

import Foundation
import MLX
import MLXNN

public class YarnRoPE: Module {
    let dimensions: Int
    let traditional: Bool
    let maxPositionEmbeddings: Int
    let base: Float
    let scalingFactor: Float
    let originalMaxPositionEmbeddings: Int
    let betaFast: Float
    let betaSlow: Float
    let mscale: Float
    let mscaleAllDim: Float

    private let _mscale: Float
    private let _freqs: MLXArray

    public init(
        dimensions: Int,
        traditional: Bool = false,
        maxPositionEmbeddings: Int = 2048,
        base: Float = 10000,
        scalingFactor: Float = 1.0,
        originalMaxPositionEmbeddings: Int = 4096,
        betaFast: Float = 32,
        betaSlow: Float = 1,
        mscale: Float = 1,
        mscaleAllDim: Float = 0
    ) {
        precondition(dimensions % 2 == 0, "Dimensions must be even")

        self.dimensions = dimensions
        self.traditional = traditional
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base
        self.scalingFactor = scalingFactor
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.betaFast = betaFast
        self.betaSlow = betaSlow
        self.mscale = mscale
        self.mscaleAllDim = mscaleAllDim

        func yarnFindCorrectionDim(numRotations: Float) -> Float {
            return Float(dimensions)
                * log(Float(originalMaxPositionEmbeddings) / (numRotations * 2 * Float.pi))
                / (2 * log(base))
        }

        func yarnFindCorrectionRange() -> (low: Int, high: Int) {
            let low = Int(floor(yarnFindCorrectionDim(numRotations: betaFast)))
            let high = Int(ceil(yarnFindCorrectionDim(numRotations: betaSlow)))
            return (max(low, 0), min(high, dimensions - 1))
        }

        func yarnGetMscale(scale: Float, mscale: Float) -> Float {
            if scale <= 1 {
                return 1.0
            }
            return 0.1 * mscale * log(scale) + 1.0
        }

        func yarnLinearRampMask(minVal: Float, maxVal: Float, dim: Int) -> MLXArray {
            var maxVal = maxVal
            if minVal == maxVal {
                maxVal += 0.001
            }

            let linearFunc = (MLXArray(0 ..< dim).asType(.float32) - minVal) / (maxVal - minVal)
            return clip(linearFunc, min: 0, max: 1)
        }

        self._mscale =
            yarnGetMscale(scale: scalingFactor, mscale: mscale)
            / yarnGetMscale(scale: scalingFactor, mscale: mscaleAllDim)

        let freqExtra = pow(
            base,
            MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                / dimensions)
        let freqInter =
            scalingFactor
            * pow(
                base,
                MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                    / dimensions)

        let (low, high) = yarnFindCorrectionRange()
        let freqMask =
            1.0 - yarnLinearRampMask(minVal: Float(low), maxVal: Float(high), dim: dimensions / 2)

        self._freqs = (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (1 - freqMask))
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        var x = x

        if _mscale != 1.0 {
            x[.ellipsis, 0 ..< dimensions] = _mscale * x[.ellipsis, 0 ..< dimensions]
        }

        return MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: self._freqs
        )
    }
}
