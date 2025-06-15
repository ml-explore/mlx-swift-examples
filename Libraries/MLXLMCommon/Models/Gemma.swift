//
//  Gemma.swift
//  mlx-swift-examples
//
//  Created by Anthony DePasquale on 17.03.2025.
//

import Foundation
import MLX
import MLXFast
import MLXNN

public enum Gemma {
    /// Specialized norm for gemma
    public class RMSNorm: Module, UnaryLayer {
        let weight: MLXArray
        let eps: Float

        public init(dimensions: Int, eps: Float = 1e-5) {
            self.weight = MLXArray.ones([dimensions])
            self.eps = eps
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            return MLXFast.rmsNorm(x, weight: 1.0 + self.weight, eps: self.eps)
        }
    }

    /// Clips residual connections to prevent overflow in float16 operations
    static public func clipResidual(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
        if x.dtype != .float16 {
            return x + y
        }
        // IEEE 754 half-precision maximum finite value
        let bound: Float = 65504.0  // Float16 maximum finite value
        let xFloat32 = x.asType(.float32)
        let yFloat32 = y.asType(.float32)
        let result = xFloat32 + yFloat32
        return clip(result, min: MLXArray(-bound), max: MLXArray(bound)).asType(.float16)
    }
}
