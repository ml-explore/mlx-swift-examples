// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import Tokenizers

/// Protocol for models that can have LoRA adapters applied.
///
/// This protocol describes which layers (typically Attention) can
/// have adapters applied.
///
/// For example the ``LlamaModel``  has `layers` (`TransformerBlock`)
/// that each have `attention` -- these are the layers that should have the
/// adapters applied:
///
/// ```swift
/// extension LlamaModel: LoRAModel {
///     public func loraLayers() -> [LoRALayer] {
///         model.layers.map { $0.attention }
///     }
/// }
/// ```
///
/// ### See Also
/// - ``LoRALayer``
/// - ``LoRATrain/convert(model:layers:)``
public protocol LoRAModel: Module, LLMModel {

    /// The layers that should have the LoRA adapter applied
    func loraLayers() -> [LoRALayer]
}

/// Protocol for layers that should have LoRA adapters applied.
///
/// The ``loraLinearModules()``  method should return the module names
/// and modules that should have apapters applied.  For example:
///
/// ```swift
/// extension Attention: LoRALayer {
///     func loraLinearModules() -> [String: any LoRAReplacableLinear] {
///         [
///             "q_proj": wq,
///             "v_proj": wv,
///         ]
///     }
/// }
/// ```
///
/// The properties for the layers that need adapters must be of type ``LoRAReplacableLinear``:
///
/// ```swift
/// @ModuleInfo(key: "q_proj") var wq: LoRAReplacableLinear
/// @ModuleInfo(key: "k_proj") var wk: Linear
/// @ModuleInfo(key: "v_proj") var wv: LoRAReplacableLinear
/// @ModuleInfo(key: "o_proj") var wo: Linear
/// ```
///
/// This is required so that the property can hold either a `Linear` layer or a ``LoRALinear``
/// layer.
///
/// ### See Also
/// - ``LoRAModel``
/// - ``LoRATrain/convert(model:layers:)``
public protocol LoRALayer: Module {

    /// Return the `Linear` layers that should have ``LoRALinear`` applied.
    func loraLinearModules() -> [String: LoRAReplacableLinear]
}

/// Type that allows properties to hold either `Linear` or ``LoRALinear`` values.
///
/// ### See Also
/// - ``LoRALayer``
public protocol LoRAReplacableLinear: Module, UnaryLayer {

}

extension Linear: LoRAReplacableLinear {
}

/// Implementation of LoRA `Linear` replacement layer.
///
/// This layer implements the LoRA capabilities, specifically:
///
/// - converting `Linear` or `QuantizedLinear` layers to ``LoRALinear``
/// - converting ``LoRALinear`` back to `Linear` or `QuantizedLinear`
/// - implementing the LoRA evaluation
///
/// This is not typically used directly -- ``LoRATrain/convert(model:layers:)`` is used to
/// add the adapter layers to a given model.
///
/// ### See Also
/// - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
/// - [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
/// - ``LoRALayer``
/// - ``LoRATrain/convert(model:layers:)``
/// - ``LoRATrain/fuse(model:deQuantize:)``
public class LoRALinear: Module, LoRAReplacableLinear {

    let scale: Float

    @ModuleInfo var linear: Linear

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    public init(
        _ inputDimensions: Int, _ outputDimensions: Int, rank: Int = 8, bias: Bool = false,
        scale: Float = 20.0, linear: Linear? = nil
    ) {

        // Scale for low-rank update
        self.scale = scale

        // Low rank lora weights
        let loraScale = 1 / sqrt(Float(inputDimensions))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

        self.linear = linear ?? Linear(inputDimensions, outputDimensions, bias: bias)
    }

    /// Convert a `Linear` or `QuantizedLinear` layer into ``LoRALinear``.
    ///
    /// This is typically called via ``LoRATrain/convert(model:layers:)``.
    ///
    /// ### See Also
    /// - ``LoRATrain/convert(model:layers:)``
    public static func from(linear: Linear, rank: Int = 8) -> LoRALinear {
        var (outputDimensions, inputDimensions) = linear.shape
        if let l = linear as? QuantizedLinear {
            inputDimensions = inputDimensions * 32 / l.bits
        }
        return LoRALinear(inputDimensions, outputDimensions, rank: rank, linear: linear)
    }

    var dtype: DType {
        let dtype: DType
        if let q = linear as? QuantizedLinear {
            dtype = q.scales.dtype
        } else {
            dtype = linear.weight.dtype
        }
        return dtype
    }

    /// Convert a ``LoRALinear`` back into a fused `Linear` or `QuantizedLinear` layer.
    ///
    /// This is typically called via ``LoRATrain/fuse(model:deQuantize:)``.
    ///
    /// ### See Also
    /// - ``LoRATrain/fuse(model:deQuantize:)``
    public func toLinear(deQuantize: Bool = false) -> Linear {
        var weight = linear.weight
        var dtype = weight.dtype

        if let q = linear as? QuantizedLinear {
            dtype = .float16
            weight = dequantized(
                weight, scales: q.scales, biases: q.biases, groupSize: q.groupSize, bits: q.bits)
        }

        let (outputDimensions, inputDimensions) = linear.shape
        var fusedLinear = Linear(inputDimensions, outputDimensions, bias: linear.bias != nil)

        let loraB = (scale * loraB.T).asType(dtype)
        let loraA = loraA.T.asType(dtype)
        var parameters = ModuleParameters()
        parameters["weight"] = .value(weight + matmul(loraB, loraA))
        if let bias = linear.bias {
            parameters["bias"] = .value(bias)
        }
        fusedLinear.update(parameters: parameters)

        if !deQuantize, let q = linear as? QuantizedLinear {
            fusedLinear = QuantizedLinear.from(
                linear: fusedLinear, groupSize: q.groupSize, bits: q.bits)
        }

        return fusedLinear
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = linear(x.asType(self.dtype))
        let z = matmul(matmul(x, self.loraA), self.loraB)
        return y + scale * z
    }
}

/// Equivalent to `lora.py/iterate_batches()`
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

public enum LoRATrain {

    public typealias LoraLossFunction = (Module, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray
    )

    /// LoRA training parameters
    public struct Parameters {
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

        public init() {
        }
    }

    /// Freeze the model layers and replace the indicated modules (Linear) that should be
    /// converted to ``LoRALayer`` and remain trainable.
    ///
    /// Once a model has had the LoRA adapters applied, adapter weights can be loaded
    /// (if available):
    ///
    /// ```swift
    /// try LoRATrain.loadLoRAWeights(model: model, url: args.adapter)
    /// ```
    ///
    /// At this point the model is ready for one or more of the following:
    ///
    /// - training with ``train(model:train:validate:optimizer:loss:tokenizer:parameters:progress:)``
    /// - loss evaluation with ``evaluate(model:dataset:loss:tokenizer:batchSize:batchCount:)``
    /// - fusing with ``fuse(model:deQuantize:)``
    /// - text generation with ``generate(promptTokens:parameters:model:tokenizer:didGenerate:)``
    ///     - note that this is just using normal model text generation
    ///
    /// - Parameters:
    ///   - model: model to convert
    ///   - layers: number of suffix layers to convert
    public static func convert(model: LoRAModel, layers: Int) {
        model.freeze()

        for layer in model.loraLayers().suffix(layers) {
            var update = ModuleChildren()
            for (key, linear) in layer.loraLinearModules() {
                if let linear = linear as? Linear {
                    update[key] = .value(LoRALinear.from(linear: linear))
                }
            }
            layer.update(modules: update)
        }
    }

    /// Fuses the LoRA adapters back into the model weights.
    ///
    /// This produces a model in the original format with `Linear` or `QuantizedLinear` layer
    /// weights that incorporate the LoRA adapter.
    ///
    /// - Parameters:
    ///   - model: model to convert
    ///   - deQuantize: if `true` will convert `QuantizedLinear` back into `Linear`
    public static func fuse(model: LoRAModel, deQuantize: Bool = false) {
        for layer in model.loraLayers() {
            var update = ModuleChildren()
            for (key, linear) in layer.loraLinearModules() {
                if let lora = linear as? LoRALinear {
                    update[key] = .value(lora.toLinear(deQuantize: deQuantize))
                }
            }
            if !update.isEmpty {
                layer.update(modules: update)
            }
        }
    }

    public static func loss(model: Module, inputs: MLXArray, targets: MLXArray, lengths: MLXArray)
        -> (
            MLXArray, MLXArray
        )
    {
        // def loss(model, inputs, targets, lengths):

        // run model on inputs
        let model = model as! LLMModel
        let logits = model(inputs, cache: nil).0.asType(.float32)

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

    /// Given a ``LoRAModel`` with LoRA adaptors applied, load adapter weights from a `.safetensors` file.
    public static func loadLoRAWeights(model: LoRAModel, url: URL) throws {
        let weights = try ModuleParameters.unflattened(loadArrays(url: url))
        try model.update(parameters: weights, verify: .noUnusedKeys)
        eval(model)
    }

    /// Given a ``LoRAModel`` with LoRA adaptors applied, write adapter weights to a `.safetensors` file.
    public static func saveLoRAWeights(model: LoRAModel, url: URL) throws {
        let parameters = Dictionary(
            uniqueKeysWithValues: model.trainableParameters().flattened())
        try save(arrays: parameters, url: url)
    }

    public enum Progress: CustomStringConvertible {
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

    public enum ProgressDisposition {
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
        model: LoRAModel, train: [String], validate: [String], optimizer: Optimizer,
        loss: @escaping LoraLossFunction = loss, tokenizer: Tokenizer, parameters: Parameters,
        progress: (Progress) async -> ProgressDisposition
    ) async throws {
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

                if await progress(
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

                if await progress(
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

                if await progress(.save(iteration: iteration, url: adapterURL)) == .stop {
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
