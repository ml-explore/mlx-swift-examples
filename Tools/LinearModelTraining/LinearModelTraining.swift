// Copyright © 2024 Apple Inc.

import ArgumentParser
import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

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
        // for a linear function: y = mx + b.  This can be trained
        // to match data -- in this case an unknown (to the model)
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

        // measure the distance from the prediction (model(x)) and the
        // ground truth (y).  this gives feedback on how close the
        // prediction is from matching the truth
        func loss(model: LinearFunctionModel, x: MLXArray, y: MLXArray) -> MLXArray {
            mseLoss(predictions: model(x), targets: y, reduction: .mean)
        }

        let model = LinearFunctionModel()
        eval(model.parameters())

        let lg = valueAndGrad(model: model, loss)

        // the optimizer will use the gradients update the model parameters
        let optimizer = SGD(learningRate: 1e-1)

        // the function to train our model against -- it doesn't have
        // to be linear, but matching what the model models is easy
        // to understand
        func f(_ x: MLXArray) -> MLXArray {
            // these are the target parameters
            let m = self.m
            let b = self.b

            // our actual function
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
            // we expect that the parameters will approach the targets
            print("target: b = \(b), m = \(m)")
            print("parameters: \(model.parameters())")

            // generate random training data along with the ground truth.
            // notice that the shape is [B, 1] where B is the batch
            // dimension -- this allows us to train on several samples simultaneously
            //
            // note: a very large batch size will take longer to converge because
            // the gradient will be representing too many samples down into
            // a single float parameter.
            let x = MLXRandom.uniform(low: -5.0, high: 5.0, [batchSize, 1])
            let y = f(x)
            eval(x, y)

            // compute the loss and gradients.  use the optimizer
            // to adjust the parameters closer to the target
            let loss = resolvedStep(x, y)

            eval(model, optimizer)

            // we should see this converge toward 0
            print("loss: \(loss)")
        }

    }
}
