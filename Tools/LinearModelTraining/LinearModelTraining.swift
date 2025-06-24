// Copyright © 2024 Apple Inc.

import ArgumentParser
import Foundation
import MLX
import MLXNN
import MLXOptimizers

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

@main
struct Train: AsyncParsableCommand {

    @Option var epochs = 20
    @Option var batchSize = 8

    @Option var m: Float = 0.25
    @Option var b: Float = 7

    @Flag var compile = false

    @Option var device = DeviceType.cpu

    func run() async throws {
        Device.setDefault(device: Device(device))

        // A very simple model that implements the equation
        // for a linear function: y = mx + b. This can be trained
        // to match data – in this case, an unknown (to the model)
        // linear function.
        //
        // This is a nice example because most people know how
        // linear functions work and we can see how the slope
        // and intercept converge.
        class LinearFunctionModel: Module, UnaryLayer {
            let m = MLXRandom.uniform(low: -5.0, high: 5.0)
            let b = MLXRandom.uniform(low: -5.0, high: 5.0)

            func callAsFunction(_ x: MLXArray) -> MLXArray {
                m * x + b
            }
        }

        // Measure the distance from the prediction (model(x)) and the
        // ground truth (y). This gives feedback on how close the
        // prediction is from matching the truth.
        func loss(model: LinearFunctionModel, x: MLXArray, y: MLXArray) -> MLXArray {
            mseLoss(predictions: model(x), targets: y, reduction: .mean)
        }

        let model = LinearFunctionModel()
        eval(model.parameters())

        let lg = valueAndGrad(model: model, loss)

        // The optimizer will use the gradients update the model parameters
        let optimizer = SGD(learningRate: 1e-1)

        // The function to train our model against. It doesn't have
        // to be linear, but matching what the model models is easy
        // to understand.
        func f(_ x: MLXArray) -> MLXArray {
            // These are the target parameters
            let m = self.m
            let b = self.b

            // Our actual function
            return m * x + b
        }

        func step(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
            let (loss, grads) = lg(model, x, y)
            optimizer.update(model: model, gradients: grads)
            return loss
        }

        let resolvedStep =
            self.compile
            ? MLX.compile(inputs: [model, optimizer], outputs: [model, optimizer], step) : step

        for _ in 0 ..< epochs {
            // We expect that the parameters will approach the targets
            print("target: b = \(b), m = \(m)")
            print("parameters: \(model.parameters())")

            // Generate random training data along with the ground truth.
            // Notice that the shape is [B, 1] where B is the batch
            // dimension. This allows us to train on several samples simultaneously.
            //
            // Note: A very large batch size will take longer to converge because
            // the gradient will be representing too many samples down into
            // a single float parameter.
            let x = MLXRandom.uniform(low: -5.0, high: 5.0, [batchSize, 1])
            let y = f(x)
            eval(x, y)

            // Compute the loss and gradients. Use the optimizer
            // to adjust the parameters closer to the target.
            let loss = resolvedStep(x, y)

            eval(model, optimizer)

            // We should see this converge toward 0
            print("loss: \(loss)")
        }

    }
}
