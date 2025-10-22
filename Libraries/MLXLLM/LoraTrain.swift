// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXOptimizers
import Tokenizers

/// Equivalent to `lora.py/iterate_batches()`. Used internally by ``LoRATrain``.
struct LoRABatchIterator: Sequence, IteratorProtocol {

    let dataset: [String]
    let batchSize: Int
    let tokenizer: Tokenizer

    let train: Bool

    var indices: [Int]
    var index = 0

    public init(dataset: [String], tokenizer: Tokenizer, batchSize: Int, train: Bool) {
        self.dataset = dataset
        self.batchSize = batchSize
        self.tokenizer = tokenizer
        self.train = train

        self.indices = Array(0 ..< dataset.count)
        if train {
            indices.shuffle()
        }
    }

    mutating public func next() -> (MLXArray, MLXArray, MLXArray)? {
        if index >= indices.count {
            if !train {
                return nil
            }

            indices.shuffle()
            index = 0
        }

        let endIndex = Swift.min(index + batchSize, indices.count)

        let batch = (index ..< endIndex)
            .map { tokenizer.encode(text: dataset[indices[$0]]) }
        let lengths = batch.map { $0.count }
        let maxLength = lengths.max() ?? 0

        if maxLength > 2048 {
            print(
                """
                [WARNING] Some sequences are longer than 2048 tokens.
                Consider pre-splitting your data to save memory.
                """)
        }

        // pad to the max length
        let batchArray = MLXArray.zeros([lengths.count, maxLength], type: Int32.self)
        for (j, (b, l)) in zip(batch, lengths).enumerated() {
            batchArray[j, 0 ..< l] = MLXArray(b)
        }

        index = endIndex

        return (batchArray[0..., .stride(to: -1)], batchArray[0..., 1...], MLXArray(lengths))
    }
}

/// Collection of functions for adding LoRA adapters to an LLM model, training, fusing and saving/loading weights.
///
/// The typical flow for training is:
///
/// ```swift
/// // load the base model and tokenizer
/// let (model, tokenizer) = try await LLM.load(configuration: ModelConfiguration.mistral7B4bit)
///
/// // add LoRALinear adapter layers
/// LoRATrain.convert(model: model, layers: Array(model.loraLinearLayers().suffix(4)))
///
/// // optionally load LoRA weights
/// try LoRATrain.loadLoRAWeights(model: model, url: ...)
///
/// // load the train/validation data
/// let train = try loadLoRAData(directory: data, name: "train")
/// let valid = try loadLoRAData(directory: data, name: "valid")
///
/// // train
/// let optimizer = Adam(learningRate: 1e-5)
/// try await LoRATrain.train(
///     model: model, train: train, validate: valid, optimizer: optimizer, tokenizer: tokenizer,
///     parameters: LoRATrain.Parameters()
/// ) { progress in
///     print(progress)
///     return .more
/// }
/// ```
///
/// At this point the model will be trained and you could do one of the following:
///
/// - ``saveLoRAWeights(model:url:)`` -- write the LoRA weights to a file
/// - ``fuse(model:layers:deQuantize:)`` -- fuse the LoRA weights and convert back into the original model
///     architecture. These weights can be saved and reloaded with normal model handling code.
/// - ``evaluate(model:dataset:loss:tokenizer:batchSize:batchCount:)``-- compute the test loss
///     againts a test dataset
/// - use the in memory model as a normal `LLMModel` and evaluate a prompt
///
public enum LoRATrain {

    public typealias LoraLossFunction = (Module, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray
    )

    /// LoRA training parameters
    public struct Parameters: Sendable {
        /// number of prompts to evaluate per iteration
        public var batchSize = 4

        /// number of iterations to train for
        public var iterations = 1000

        /// number of training steps between loss reporting
        public var stepsPerReport = 10

        /// number of steps between validations
        public var stepsPerEval = 100

        /// number of validations batches, `0` uses the entire validation set
        public var validationBatches = 10

        /// save the model every N iterations
        public var saveEvery = 100

        /// save path for the adapter `.safetensors`
        public var adapterURL: URL?

        public init(
            batchSize: Int = 4, iterations: Int = 1000, stepsPerReport: Int = 10,
            stepsPerEval: Int = 100, validationBatches: Int = 10, saveEvery: Int = 100,
            adapterURL: URL? = nil
        ) {
            self.batchSize = batchSize
            self.iterations = iterations
            self.stepsPerReport = stepsPerReport
            self.stepsPerEval = stepsPerEval
            self.validationBatches = validationBatches
            self.saveEvery = saveEvery
            self.adapterURL = adapterURL
        }
    }

    public static func loss(model: Module, inputs: MLXArray, targets: MLXArray, lengths: MLXArray)
        -> (
            MLXArray, MLXArray
        )
    {
        // def loss(model, inputs, targets, lengths):

        // run model on inputs
        let model = model as! any LLMModel
        let logits = model(inputs, cache: nil).asType(.float32)

        // mask padding tokens
        let lengthMask = MLXArray(0 ..< inputs.dim(1))[.newAxis, 0...] .< lengths[0..., .newAxis]

        // calculate the loss
        let ntoks = lengthMask.sum()
        let ce = (crossEntropy(logits: logits, targets: targets) * lengthMask).sum() / ntoks
        return (ce, ntoks)
    }

    /// Evaluate the model and dataset and return the loss over the entire dataset.
    ///
    /// - Parameters:
    ///   - model: the model to evaluate
    ///   - dataset: the dataset
    ///   - loss: loss function
    ///   - tokenizer: tokenizer
    ///   - batchSize: number of items from the dataset to evaluate at once
    ///   - batchCount: number of batch elements to evaluate, 0 for all
    /// - Returns: the loss over the enumerate data
    ///
    /// ### See Also
    /// - ``loadLoRAData(directory:name:)``
    public static func evaluate(
        model: Module, dataset: [String], loss: LoraLossFunction = loss, tokenizer: Tokenizer,
        batchSize: Int, batchCount: Int
    ) -> Float {
        var allLosses = [Float]()
        var tokenCount = 0

        for (iteration, (inputs, targets, lengths)) in LoRABatchIterator(
            dataset: dataset, tokenizer: tokenizer, batchSize: batchSize, train: false
        ).enumerated() {
            let (losses, tokens) = loss(model, inputs, targets, lengths)
            allLosses.append((losses * tokens).item(Float.self))
            tokenCount += tokens.item(Int.self)

            if batchCount != 0 && iteration + 1 >= batchCount {
                break
            }
        }

        return (sum(MLXArray(allLosses), stream: .cpu) / tokenCount).item(Float.self)
    }

    /// Given a model with LoRA adaptors applied, write adapter weights to a `.safetensors` file.
    ///
    /// ### See Also
    /// - ``convert(model:layers:)``
    /// - ``loadLoRAWeights(model:url:)``
    public static func saveLoRAWeights(model: Module, url: URL) throws {
        let parameters = Dictionary(
            uniqueKeysWithValues: model.trainableParameters().flattened())
        try save(arrays: parameters, url: url)
    }

    public enum Progress: CustomStringConvertible, Sendable {
        case train(
            iteration: Int, trainingLoss: Float, iterationsPerSecond: Double,
            tokensPerSecond: Double)
        case validation(iteration: Int, validationLoss: Float, validationTime: Double)
        case save(iteration: Int, url: URL)

        public var description: String {
            switch self {
            case .train(
                let iteration, let trainingLoss, let iterationsPerSecond, let tokensPerSecond):
                "Iteration \(iteration + 1): training loss \(trainingLoss.formatted()), "
                    + "iterations/sec \(iterationsPerSecond.formatted()), "
                    + "Tokens/sec \(tokensPerSecond.formatted())"
            case .validation(let iteration, let validationLoss, let validationTime):
                "Iteration \(iteration + 1): "
                    + "validation loss \(validationLoss.formatted()), "
                    + "validation time \(validationTime.formatted())s"
            case .save(let iteration, let url):
                "Iteration \(iteration + 1): saved weights to \(url.path())"
            }
        }
    }

    public enum ProgressDisposition: Sendable {
        case stop
        case more
    }

    /// Train (or continue training) LoRA weights.
    ///
    /// - Parameters:
    ///   - model: model to train
    ///   - train: training dataset
    ///   - validate: validate dataset
    ///   - optimizer: optimizer used in training
    ///   - loss: loss function
    ///   - tokenizer: tokenizer
    ///   - parameters: training parameters
    ///   - progress: progress callback
    public static func train(
        model: Module, train: [String], validate: [String], optimizer: Optimizer,
        loss: @escaping LoraLossFunction = loss, tokenizer: Tokenizer, parameters: Parameters,
        progress: (Progress) -> ProgressDisposition
    ) throws {
        // def train(model, train_set, val_set, optimizer, loss, tokenizer, args)

        let lossValueGrad = valueAndGrad(model: model) { model, arrays in
            let (ce, ntoks) = loss(model, arrays[0], arrays[1], arrays[2])
            return [ce, ntoks]
        }

        var losses = [Float]()
        var tokenCount = 0

        var start = Date.timeIntervalSinceReferenceDate

        for (iteration, (inputs, targets, lengths)) in LoRABatchIterator(
            dataset: train, tokenizer: tokenizer, batchSize: parameters.batchSize, train: true
        ).enumerated() {
            // forward and backward pass
            let (resultArray, grad) = lossValueGrad(model, [inputs, targets, lengths])
            let lvalue = resultArray[0]
            let tokens = resultArray[1]

            // model update
            optimizer.update(model: model, gradients: grad)
            eval(model, optimizer, lvalue)

            // record loss
            losses.append(lvalue.item(Float.self))
            tokenCount += tokens.item(Int.self)

            // report training loss
            if (iteration + 1) % parameters.stepsPerReport == 0 {
                let trainingLoss = MLXArray(losses).mean(stream: .cpu).item(Float.self)
                let now = Date.timeIntervalSinceReferenceDate

                let iterationsPerSecond = Double(parameters.stepsPerReport) / (now - start)
                let tokensPerSecond = Double(tokenCount) / (now - start)

                if progress(
                    .train(
                        iteration: iteration, trainingLoss: trainingLoss,
                        iterationsPerSecond: iterationsPerSecond, tokensPerSecond: tokensPerSecond))
                    == .stop
                {
                    break
                }

                losses.removeAll()
                tokenCount = 0
                start = Date.timeIntervalSinceReferenceDate
            }

            // report validation loss
            if iteration == 0 || (iteration + 1) % parameters.stepsPerEval == 0 {
                let validationStart = Date.timeIntervalSinceReferenceDate
                let validationLoss = evaluate(
                    model: model, dataset: validate, loss: loss, tokenizer: tokenizer,
                    batchSize: parameters.batchSize, batchCount: parameters.validationBatches)
                let now = Date.timeIntervalSinceReferenceDate

                if progress(
                    .validation(
                        iteration: iteration, validationLoss: validationLoss,
                        validationTime: now - validationStart)) == .stop
                {
                    break
                }

                start = Date.timeIntervalSinceReferenceDate
            }

            // save adapter weights if needed
            if let adapterURL = parameters.adapterURL, (iteration + 1) % parameters.saveEvery == 0 {
                try saveLoRAWeights(model: model, url: adapterURL)

                if progress(.save(iteration: iteration, url: adapterURL)) == .stop {
                    break
                }

                start = Date.timeIntervalSinceReferenceDate
            }

            if iteration + 1 >= parameters.iterations {
                break
            }
        }
    }
}
