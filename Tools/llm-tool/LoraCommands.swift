// Copyright © 2024 Apple Inc.

import ArgumentParser
import Foundation
import Hub
import LLM
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import Tokenizers

struct LoRACommand: AsyncParsableCommand {

    static var configuration = CommandConfiguration(
        commandName: "lora",
        abstract: "LoRA commands",
        subcommands: [
            LoRATrainCommand.self, LoRAFuseCommand.self, LoRATestCommand.self, LoRAEvalCommand.self,
        ]
    )
}

/// Common arguments for loading a LoRA mdoel with adapter weights
struct LoRAModelArguments: ParsableArguments {

    @OptionGroup var args: ModelArguments

    @Option(name: .long, help: "Save/load path for the trained adapter weights")
    public var adapter: URL = URL(filePath: "adapters.safetensors")

    @Option(name: .long, help: "Number of layers to fine-tune")
    public var loraLayers = 16

    /// Load the model and apply the LoRA adapters.
    ///
    /// This does not load the adapter weights as they may not exist yet.
    func load() async throws -> (LLMModel, Tokenizer, ModelConfiguration) {
        let (model, tokenizer, modelConfiguration) = try await args.load()

        // convert some of the Linear layers to LoRALinear
        LoRATrain.convert(model: model, layers: loraLayers(model: model))

        return (model, tokenizer, modelConfiguration)
    }

    func loraLayers(model: Module) -> LoRALinearLayers {
        guard let layerProvider = model as? LoRAModel else {
            // the layerProvider will indicate which Linear layers need to be replaced
            fatalError(
                "Model \(type(of: model)) (\(args.model)) must implement the LoRALayerProvider protocol"
            )
        }

        return Array(layerProvider.loraLinearLayers().suffix(loraLayers))
    }

    func describe(model: Module) {
        let totalParameterCount = model.parameters()
            .flattenedValues().map { $0.size }.reduce(0, +)
        let trainableParameterCount = model.trainableParameters()
            .flattenedValues().map { $0.size }.reduce(0, +)

        print("Model: \(args.model)")
        print("Total parameters: \((totalParameterCount / 1_000_000).formatted())M")
        print(
            "Trainable parameters: \((Float(trainableParameterCount) / 1_000_000).formatted(.number.precision(.significantDigits(1 ..< 4))))M"
        )

    }
}

struct LoRATrainCommand: AsyncParsableCommand {

    static var configuration = CommandConfiguration(
        commandName: "train",
        abstract: "LoRA training"
    )

    @OptionGroup var args: LoRAModelArguments
    @OptionGroup var memory: MemoryArguments

    @Flag(help: "Resume training with the given adapter file")
    public var resume = false

    @Option(name: .long, help: "Directory with {train, valid, test}.{jsonl,txt} files")
    public var data: URL = URL(filePath: "data")

    @Option(name: .long, help: "Learning rate for the optimizer")
    public var learningRate: Float = 1e-5

    @Option(name: .long, help: "Number of dataset items to evaluate per iteration (batch)")
    public var batchSize = 4

    @Option(name: .long, help: "Number iterations to train for")
    public var iterations = 1000

    @Option(name: .long, help: "Number of iterations between loss reporting")
    public var stepsPerReport = 10

    @Option(name: .long, help: "Number of iterations between validations")
    public var stepsPerEval = 100

    @Option(name: .long, help: "Number of validation batches, 0 uses the entire set")
    public var validationBatches = 10

    @Option(name: .long, help: "Number of iterations between checkpointing the adapter weights")
    public var saveEvery = 100

    var parameters: LoRATrain.Parameters {
        var p = LoRATrain.Parameters()
        p.batchSize = self.batchSize
        p.iterations = self.iterations
        p.stepsPerReport = self.stepsPerReport
        p.stepsPerEval = self.stepsPerEval
        p.validationBatches = self.validationBatches
        p.saveEvery = self.saveEvery
        p.adapterURL = args.adapter
        return p
    }

    @MainActor
    mutating func run() async throws {
        let (model, tokenizer, _) = try await args.load()
        args.describe(model: model)

        memory.start()

        if resume {
            print("Loading pretrained adapters from \(args.adapter.path())")
            try LoRATrain.loadLoRAWeights(model: model, url: args.adapter)
        }

        // load the train/validation data
        let train = try loadLoRAData(directory: data, name: "train")
        let valid = try loadLoRAData(directory: data, name: "valid")

        if train.isEmpty {
            fatalError("Training set is empty: \(data.path()))")
        }
        if valid.isEmpty {
            fatalError("Validation set is empty: \(data.path()))")
        }

        // train
        let optimizer = Adam(learningRate: learningRate)
        try await LoRATrain.train(
            model: model, train: train, validate: valid, optimizer: optimizer, tokenizer: tokenizer,
            parameters: parameters
        ) { progress in
            print(progress)
            return .more
        }
        try LoRATrain.saveLoRAWeights(model: model, url: args.adapter)
    }
}

struct LoRAFuseCommand: AsyncParsableCommand {

    static var configuration = CommandConfiguration(
        commandName: "fuse",
        abstract: "Fuse lora adapter weights back in to original model"
    )

    @OptionGroup var args: LoRAModelArguments

    @Flag(name: .long, help: "De-quantize QuantizedLinear layers back into Linear")
    var deQuantize = false

    @Option(name: .long, help: "Hub ID (mlx-community/mistral-lora) or path (/tmp/mistral-lora)")
    var output: String

    @MainActor
    mutating func run() async throws {
        let outputURL: URL
        if output.hasPrefix("/") {
            outputURL = URL(filePath: output)
        } else {
            let repo = HubApi.Repo(id: output)
            outputURL = HubApi().localRepoLocation(repo)
        }

        let (model, _, modelConfiguration) = try await args.load()

        // load the prepared weights
        try LoRATrain.loadLoRAWeights(model: model, url: args.adapter)

        // fuse them back into Linear/QuantizedLinear
        LoRATrain.fuse(model: model, layers: args.loraLayers(model: model), deQuantize: deQuantize)

        // make the new directory and copy files from source model
        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
        let inputURL = modelConfiguration.modelDirectory()
        let enumerator = FileManager.default.enumerator(
            at: inputURL, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator {
            // copy everything except the model weights -- we will write out the fused one below
            if url.pathExtension == "safetensors" {
                continue
            }

            try FileManager.default.copyItem(
                at: url, to: outputURL.appending(component: url.lastPathComponent))
        }

        // write them back out
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try save(arrays: weights, url: outputURL.appending(component: "weights.safetensors"))

        print("Fused weights written to \(outputURL.path())")
        print("Use with:\n\tllm-tool eval --model \(output)")
    }

}

struct LoRATestCommand: AsyncParsableCommand {

    static var configuration = CommandConfiguration(
        commandName: "test",
        abstract: "LoRA testing"
    )

    @OptionGroup var args: LoRAModelArguments
    @OptionGroup var memory: MemoryArguments

    @Option(name: .long, help: "Directory with {train, valid, test}.{jsonl,txt} files")
    public var data: URL = URL(filePath: "data")

    @Option(name: .long, help: "Minibatch size")
    public var batchSize = 4

    @MainActor
    mutating func run() async throws {
        let (model, tokenizer, _) = try await args.load()
        args.describe(model: model)
        try LoRATrain.loadLoRAWeights(model: model, url: args.adapter)

        memory.start()

        let test = try loadLoRAData(directory: data, name: "test")
        let loss = LoRATrain.evaluate(
            model: model, dataset: test, tokenizer: tokenizer, batchSize: batchSize, batchCount: 0)

        print("Test loss \(loss.formatted()), ppl \(exp(loss).formatted())")
    }

}

struct LoRAEvalCommand: AsyncParsableCommand {

    static var configuration = CommandConfiguration(
        commandName: "eval",
        abstract: "LoRA evaluation"
    )

    @OptionGroup var args: LoRAModelArguments
    @OptionGroup var memory: MemoryArguments
    @OptionGroup var generate: GenerateArguments

    @MainActor
    mutating func run() async throws {
        let (model, tokenizer, modelConfiguration) = try await args.load()
        args.describe(model: model)
        try LoRATrain.loadLoRAWeights(model: model, url: args.adapter)

        memory.start()

        let (prompt, promptTokens) = try generate.tokenizePrompt(
            configuration: modelConfiguration, tokenizer: tokenizer)

        if !generate.quiet {
            print("Starting generation ...")
            print(prompt, terminator: "")
        }

        // generate and print the result
        let _ = await generate.generate(
            promptTokens: promptTokens, model: model, tokenizer: tokenizer,
            extraEOSTokens: modelConfiguration.extraEOSTokens)
        print()
    }
}
