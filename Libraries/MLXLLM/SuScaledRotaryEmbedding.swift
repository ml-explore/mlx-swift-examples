import Foundation
import MLX
import MLXFast
import MLXNN

public class SuScaledRotaryEmbedding: Module {
    let dimensions: Int
    let maxPositionEmbeddings: Int
    let originalMaxPositionEmbeddings: Int
    let scale: Float
    let _freqs: MLXArray

    public init(
        dimensions: Int,
        base: Float = 10000.0,
        maxPositionEmbeddings: Int = 131072,
        originalMaxPositionEmbeddings: Int = 4096,
        longFactor: [Float] = [1.0],
        // shortMScale: Float? = nil,
        longMScale: Float? = nil
    ) {
        precondition(dimensions % 2 == 0, "Dimensions must be even")

        self.dimensions = dimensions
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings

        let exponent =
            MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / Float(dimensions)
        let freqs = MLX.pow(MLXArray(base), exponent)
        self._freqs = MLXArray(longFactor).asType(.float32) * freqs

        self.scale =
            longMScale
            ?? sqrt(
                1 + log(Float(maxPositionEmbeddings) / Float(originalMaxPositionEmbeddings))
                    / log(Float(originalMaxPositionEmbeddings))
            )
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        // Apply scaling only to the dimensions that will be rotated
        var scaledX = x
        let sliceToScale = scaledX[.ellipsis, 0 ..< dimensions]
        scaledX[.ellipsis, 0 ..< dimensions] = scale * sliceToScale

        return MLXFast.RoPE(
            scaledX,
            dimensions: dimensions,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: self._freqs
        )
    }
}
