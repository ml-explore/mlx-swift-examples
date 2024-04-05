// Copyright © 2024 Apple Inc.

import ArgumentParser
import Foundation
import LLM
import MLX
import MLXRandom
import Tokenizers

@main
struct LLMTool: AsyncParsableCommand {
    static var configuration = CommandConfiguration(
        abstract: "Command line tool for generating text and manipulating LLMs",
        subcommands: [EvaluateCommand.self, LoRACommand.self],
        defaultSubcommand: EvaluateCommand.self)
}

/// Command line arguments for loading a model.
struct ModelArguments: ParsableArguments {

    @Option(name: .long, help: "Name of the huggingface model")
    var model: String = "mlx-community/Mistral-7B-v0.1-hf-4bit-mlx"

    @Option(name: .long, help: "Optional URL of .safetensors weights file")
    var weights: URL?

    func load() async throws -> (LLMModel, Tokenizer, ModelConfiguration) {
        let modelConfiguration = ModelConfiguration.configuration(id: model)
        let (model, tokenizer) = try await LLM.load(
            configuration: modelConfiguration, overrideWeights: weights)
        return (model, tokenizer, modelConfiguration)
    }
}

/// Command line arguments for controlling generation of text.
struct GenerateArguments: ParsableArguments {

    @Option(name: .shortAndLong, help: "The message to be processed by the model")
    var prompt = "compare python and swift"

    @Option(name: .shortAndLong, help: "Maximum number of tokens to generate")
    var maxTokens = 100

    @Option(name: .shortAndLong, help: "The sampling temperature")
    var temperature: Float = 0.6

    @Option(name: .long, help: "The top p sampling")
    var topP: Float = 0.9

    @Option(name: .long, help: "The penalty factor for repeating tokens")
    var repetitionPenalty: Float = 1.0

    @Option(name: .long, help: "The number of tokens to consider for repetition penalty")
    var repetitionContextSize: Int = 20

    @Option(name: .long, help: "The PRNG seed")
    var seed: UInt64 = 0

    var generateParameters: GenerateParameters {
        GenerateParameters(
            temperature: temperature, topP: topP, repetitionPenalty: repetitionPenalty,
            repetitionContextSize: repetitionContextSize)
    }

    func tokenizePrompt(configuration: ModelConfiguration, tokenizer: Tokenizer) -> (String, [Int])
    {
        MLXRandom.seed(seed)

        let prompt = configuration.prepare(prompt: self.prompt)
        let promptTokens = tokenizer.encode(text: prompt)

        return (prompt, promptTokens)
    }

    func generate(promptTokens: [Int], model: LLMModel, tokenizer: Tokenizer) async
        -> GenerateResult
    {
        // track how much we have printed
        var printed = 0

        return await LLM.generate(
            promptTokens: promptTokens, parameters: generateParameters,
            model: model, tokenizer: tokenizer
        ) { tokens in

            // print any new parts of the string
            let fullOutput = tokenizer.decode(tokens: tokens)
            let emitLength = fullOutput.count - printed
            let suffix = fullOutput.suffix(emitLength)
            print(suffix, terminator: "")
            fflush(stdout)

            printed = fullOutput.count

            if tokens.count >= maxTokens {
                return .stop
            } else {
                return .more
            }
        }
    }
}

/// Argument package for adjusting and reporting memory use.
struct MemoryArguments: ParsableArguments {

    @Flag(name: .long, help: "Show memory stats")
    var memoryStats = false

    @Option(name: .long, help: "Maximum cache size in M")
    var cacheSize: Int?

    @Option(name: .long, help: "Maximum memory size in M")
    var memorySize: Int?

    var startMemory: GPU.Snapshot?

    mutating func start<L>(_ load: () async throws -> L) async throws -> L {
        if let cacheSize {
            GPU.set(cacheLimit: cacheSize * 1024 * 1024)
        }

        if let memorySize {
            GPU.set(memoryLimit: memorySize * 1024 * 1024)
        }

        let result = try await load()
        startMemory = GPU.snapshot()

        return result
    }

    mutating func start() {
        if let cacheSize {
            GPU.set(cacheLimit: cacheSize * 1024 * 1024)
        }

        if let memorySize {
            GPU.set(memoryLimit: memorySize * 1024 * 1024)
        }

        startMemory = GPU.snapshot()
    }

    func reportCurrent() {
        if memoryStats {
            let memory = GPU.snapshot()
            print(memory.description)
        }
    }

    func reportMemoryStatistics() {
        if memoryStats, let startMemory {
            let endMemory = GPU.snapshot()

            print("=======")
            print("Memory size: \(GPU.memoryLimit / 1024)K")
            print("Cache size:  \(GPU.cacheLimit / 1024)K")

            print("")
            print("=======")
            print("Starting memory")
            print(startMemory.description)

            print("")
            print("=======")
            print("Ending memory")
            print(endMemory.description)

            print("")
            print("=======")
            print("Growth")
            print(startMemory.delta(endMemory).description)

        }
    }
}

struct EvaluateCommand: AsyncParsableCommand {

    static var configuration = CommandConfiguration(
        commandName: "eval",
        abstract: "evaluate prompt and generate text"
    )

    @OptionGroup var args: ModelArguments
    @OptionGroup var memory: MemoryArguments
    @OptionGroup var generate: GenerateArguments

    @MainActor
    mutating func run() async throws {
        let (model, tokenizer, modelConfiguration) = try await memory.start(args.load)

        print("Model loaded -> \(modelConfiguration.id)")

        let (prompt, promptTokens) = generate.tokenizePrompt(
            configuration: modelConfiguration, tokenizer: tokenizer)

        print("Starting generation ...")
        print(prompt, terminator: "")

        let result = await generate.generate(
            promptTokens: promptTokens, model: model, tokenizer: tokenizer)

        print()
        print("------")
        print(result.summary())

        memory.reportMemoryStatistics()
    }
}
