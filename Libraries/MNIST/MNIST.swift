// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// based on https://github.com/ml-explore/mlx-examples/blob/main/mnist/main.py

public class LeNet: Module, UnaryLayer {

    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo var pool1: MaxPool2d
    @ModuleInfo var pool2: MaxPool2d
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo var fc3: Linear

    override public init() {
        conv1 = Conv2d(inputChannels: 1, outputChannels: 6, kernelSize: 5, padding: 2)
        conv2 = Conv2d(inputChannels: 6, outputChannels: 16, kernelSize: 5, padding: 0)
        pool1 = MaxPool2d(kernelSize: 2, stride: 2)
        pool2 = MaxPool2d(kernelSize: 2, stride: 2)
        fc1 = Linear(16 * 5 * 5, 120)
        fc2 = Linear(120, 84)
        fc3 = Linear(84, 10)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        x = pool1(tanh(conv1(x)))
        x = pool2(tanh(conv2(x)))
        x = flattened(x, start: 1)
        x = tanh(fc1(x))
        x = tanh(fc2(x))
        x = fc3(x)
        return x
    }
}

public func loss(model: LeNet, x: MLXArray, y: MLXArray) -> MLXArray {
    crossEntropy(logits: model(x), targets: y, reduction: .mean)
}

public func eval(model: LeNet, x: MLXArray, y: MLXArray) -> MLXArray {
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
