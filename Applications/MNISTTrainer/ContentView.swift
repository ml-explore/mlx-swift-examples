// Copyright © 2024 Apple Inc.

import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import MNIST
import SwiftUI

struct TrainingView: View {

    @Binding var trainer: Trainer

    var body: some View {
        VStack {
            Spacer()

            ScrollView(.vertical) {
                ForEach(trainer.messages, id: \.self) {
                    Text($0)
                }
            }

            HStack {
                Spacer()
                switch trainer.state {
                case .untrained:
                    Button("Train") {
                        Task {
                            try! await trainer.run()
                        }
                    }
                case .trained(let model), .predict(let model):
                    Button("Draw a digit") {
                        trainer.state = .predict(model)
                    }
                }

                Spacer()
            }
            Spacer()
        }
        .padding()
    }
}

struct ContentView: View {
    // the training loop
    @State var trainer = Trainer()

    var body: some View {
        switch trainer.state {
        case .untrained, .trained:
            TrainingView(trainer: $trainer)
        case .predict(let model):
            PredictionView(model: model)
        }
    }
}

@Observable
class Trainer {

    enum State {
        case untrained
        case trained(LeNet)
        case predict(LeNet)
    }

    var state: State = .untrained
    var messages = [String]()

    func run() async throws {
        // Note: this is pretty close to the code in `mnist-tool`, just
        // wrapped in an Observable to make it easy to display in SwiftUI

        // download & load the training data
        let url = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        try await download(into: url)
        let data = try load(from: url)

        let trainImages = data[.init(.training, .images)]!
        let trainLabels = data[.init(.training, .labels)]!
        let testImages = data[.init(.test, .images)]!
        let testLabels = data[.init(.test, .labels)]!

        // create the model with random weights
        let model = LeNet()
        eval(model.parameters())

        // the training loop
        let lg = valueAndGrad(model: model, loss)
        let optimizer = SGD(learningRate: 0.1)

        // using a consistent random seed so it behaves the same way each time
        MLXRandom.seed(0)
        var generator: RandomNumberGenerator = SplitMix64(seed: 0)

        for e in 0 ..< 10 {
            let start = Date.timeIntervalSinceReferenceDate

            for (x, y) in iterateBatches(
                batchSize: 256, x: trainImages, y: trainLabels, using: &generator)
            {
                // loss and gradients
                let (_, grads) = lg(model, x, y)

                // use SGD to update the weights
                optimizer.update(model: model, gradients: grads)

                // eval the parameters so the next iteration is independent
                eval(model, optimizer)
            }

            let accuracy = eval(model: model, x: testImages, y: testLabels)

            let end = Date.timeIntervalSinceReferenceDate

            // add to messages -- triggers display
            await MainActor.run {
                messages.append(
                    """
                    Epoch \(e): test accuracy \(accuracy.item(Float.self).formatted())
                    Time: \((end - start).formatted())

                    """
                )
            }
        }
        await MainActor.run {
            state = .trained(model)
        }
    }
}
