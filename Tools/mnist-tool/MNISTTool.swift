// Copyright © 2024 Apple Inc.

import ArgumentParser
import Foundation
import MLX
import MLXMNIST
import MLXNN
import MLXOptimizers
import MLXRandom

@main
struct MNISTTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Command line tool for training mnist models",
        subcommands: [Train.self],
        defaultSubcommand: Train.self)
}

#if swift(>=5.10)
    extension MLX.DeviceType: @retroactive ExpressibleByArgument {
        public init?(argument: String) {
            self.init(rawValue: argument)
        }
    }
#else
    extension MLX.DeviceType: ExpressibleByArgument {
        public init?(argument: String) {
            self.init(rawValue: argument)
        }
    }
#endif

struct Train: AsyncParsableCommand {

    @Option(name: .long, help: "Directory with the training data")
    var data: String

    @Option(name: .long, help: "The PRNG seed")
    var seed: UInt64 = 0

    @Option var batchSize = 256
    @Option var epochs = 20
    @Option var learningRate: Float = 1e-1

    @Option var device = DeviceType.gpu

    @Flag var compile = false

    func run() async throws {
        Device.setDefault(device: Device(device))

        MLXRandom.seed(seed)
        var generator: RandomNumberGenerator = SplitMix64(seed: seed)

        // load the data
        let url = URL(filePath: data)

        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try await download(into: url)

        let data = try load(from: url)

        let trainImages = data[.init(.training, .images)]!
        let trainLabels = data[.init(.training, .labels)]!
        let testImages = data[.init(.test, .images)]!
        let testLabels = data[.init(.test, .labels)]!

        // create the model
        let model = LeNet()
        eval(model.parameters())

        let lg = valueAndGrad(model: model, loss)
        let optimizer = SGD(learningRate: learningRate)

        func step(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
            let (loss, grads) = lg(model, x, y)
            optimizer.update(model: model, gradients: grads)
            return loss
        }

        let resolvedStep =
            compile
            ? MLX.compile(inputs: [model, optimizer], outputs: [model, optimizer], step) : step

        for e in 0 ..< epochs {
            let start = Date.timeIntervalSinceReferenceDate

            for (x, y) in iterateBatches(
                batchSize: batchSize, x: trainImages, y: trainLabels, using: &generator)
            {
                _ = resolvedStep(x, y)

                // eval the parameters so the next iteration is independent
                eval(model, optimizer)
            }

            let accuracy = eval(model: model, x: testImages, y: testLabels)

            let end = Date.timeIntervalSinceReferenceDate

            print(
                """
                Epoch \(e): test accuracy \(accuracy.item(Float.self).formatted())
                Time: \((end - start).formatted())

                """
            )
        }
    }
}
