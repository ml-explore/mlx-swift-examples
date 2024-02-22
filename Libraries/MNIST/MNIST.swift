// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// based on https://github.com/ml-explore/mlx-examples/blob/main/mnist/main.py

public class MLP: Module, UnaryLayer {

    @ModuleInfo var layers: [Linear]

    public init(layers: Int, inputDimensions: Int, hiddenDimensions: Int, outputDimensions: Int) {
        let layerSizes =
            [inputDimensions] + Array(repeating: hiddenDimensions, count: layers) + [
                outputDimensions
            ]

        self.layers = zip(layerSizes.dropLast(), layerSizes.dropFirst())
            .map {
                Linear($0, $1)
            }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for l in layers.dropLast() {
            x = relu(l(x))
        }
        return layers.last!(x)
    }
}

public func loss(model: MLP, x: MLXArray, y: MLXArray) -> MLXArray {
    crossEntropy(logits: model(x), targets: y, reduction: .mean)
}

public func eval(model: MLP, x: MLXArray, y: MLXArray) -> MLXArray {
    mean(argMax(model(x), axis: 1) .== y)
}

private struct BatchSequence: Sequence, IteratorProtocol {

    let batchSize: Int
    let x: MLXArray
    let y: MLXArray

    let indexes: MLXArray
    var index = 0

    init(batchSize: Int, x: MLXArray, y: MLXArray, using generator: inout any RandomNumberGenerator)
    {
        self.batchSize = batchSize
        self.x = x
        self.y = y
        self.indexes = MLXArray(Array(0 ..< y.size).shuffled(using: &generator))
    }

    mutating func next() -> (MLXArray, MLXArray)? {
        guard index < y.size else { return nil }

        let range = index ..< Swift.min(index + batchSize, y.size)
        index += batchSize
        let ids = indexes[range]
        return (x[ids], y[ids])
    }
}

public func iterateBatches(
    batchSize: Int, x: MLXArray, y: MLXArray, using generator: inout any RandomNumberGenerator
) -> some Sequence<(MLXArray, MLXArray)> {
    BatchSequence(batchSize: batchSize, x: x, y: y, using: &generator)
}
