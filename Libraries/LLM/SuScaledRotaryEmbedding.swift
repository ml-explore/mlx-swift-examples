import Foundation
import MLX
import MLXFast
import MLXNN

public class SuScaledRotaryEmbedding: Module {
    let dimensions: Int
    let base: Float
    let maxPositionEmbeddings: Int
    let originalMaxPositionEmbeddings: Int
    let scale: Float
    let _freqs: MLXArray

    public init(
        dimensions: Int,
        base: Float = 10000.0,
        maxPositionEmbeddings: Int = 131072,
        originalMaxPositionEmbeddings: Int = 4096,
        longFactor: [Float] = [1.0]
    ) {
        precondition(dimensions % 2 == 0, "Dimensions must be even")

        self.dimensions = dimensions
        self.base = base
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings

        let exponent =
            MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / Float(dimensions)
        let freqs = MLX.pow(MLXArray(base), exponent)
        self._freqs = MLXArray(longFactor).asType(.float32) * freqs

        self.scale = sqrt(
            1 + log(Float(maxPositionEmbeddings) / Float(originalMaxPositionEmbeddings))
                / log(Float(originalMaxPositionEmbeddings))
        )
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        return MLXFast.RoPE(
            self.scale * x,
            dimensions: x.shape.last!,
            traditional: false,
            base: self.base,  // TODO: After updating to MLX 0.17.0, use `nil`
            scale: 1.0,
            offset: offset
                // TODO: After updating to MLX 0.17.0, pass `self._freqs` to `freqs`
        )
    }
}
