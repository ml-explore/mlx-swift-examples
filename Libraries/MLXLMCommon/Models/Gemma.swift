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
}
