// Copyright © 2024 Apple Inc.

import ArgumentParser
import Foundation
import LLM
import MLX
import MLXRandom
import Tokenizers

@main
struct LLMTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Command line tool for generating text and manipulating LLMs",
        subcommands: [EvaluateCommand.self, LoRACommand.self],
        defaultSubcommand: EvaluateCommand.self)
}

/// Command line arguments for loading a model.
struct ModelArguments: ParsableArguments, Sendable {

    @Option(name: .long, help: "Name of the huggingface model or absolute path to directory")
    var model: String = "mlx-community/Mistral-7B-v0.1-hf-4bit-mlx"

    @Sendable
    func load() async throws -> (ModelContainer, ModelConfiguration) {
        let modelConfiguration: ModelConfiguration

        if self.model.hasPrefix("/") {
            // path
            modelConfiguration = ModelConfiguration(directory: URL(filePath: self.model))
        } else {
            // identifier
            modelConfiguration = await ModelConfiguration.configuration(id: model)
        }
        let modelContainer = try await LLM.loadModelContainer(configuration: modelConfiguration)
        return (modelContainer, modelConfiguration)
    }
}

/// Command line arguments for controlling generation of text.
struct GenerateArguments: ParsableArguments, Sendable {

    @Option(
        name: .shortAndLong,
        help:
            "The message to be processed by the model.  Use @path,@path to load from files, e.g. @/tmp/prompt.txt"
    )
    var prompt: String?

    @Option(name: .shortAndLong, help: "Maximum number of tokens to generate")
    var maxTokens = 100

    @Option(name: .shortAndLong, help: "The sampling temperature")
    var temperature: Float = 0.6

    @Option(name: .long, help: "The top p sampling")
    var topP: Float = 1.0

    @Option(name: .long, help: "The penalty factor for repeating tokens")
    var repetitionPenalty: Float?

    @Option(name: .long, help: "The number of tokens to consider for repetition penalty")
    var repetitionContextSize: Int = 20

    @Option(name: .long, help: "The PRNG seed")
    var seed: UInt64 = 0

    @Flag(name: .shortAndLong, help: "If true only print the generated output")
    var quiet = false

    var generateParameters: GenerateParameters {
        GenerateParameters(
            temperature: temperature, topP: topP, repetitionPenalty: repetitionPenalty,
            repetitionContextSize: repetitionContextSize)
    }

    func resolvePrompt(configuration: ModelConfiguration) throws -> String {
        let prompt = self.prompt ?? configuration.defaultPrompt
        if prompt.hasPrefix("@") {
            let names = prompt.split(separator: ",").map { String($0.dropFirst()) }
            return try names.map { try String(contentsOfFile: $0) }.joined(separator: "\n")
        } else {
            return prompt
        }
    }

    func tokenizePrompt(configuration: ModelConfiguration, tokenizer: Tokenizer) throws -> (
        String, [Int]
    ) {
        MLXRandom.seed(seed)

        let prompt = try resolvePrompt(configuration: configuration)
        let preparedPrompt = configuration.prepare(prompt: prompt)
        let promptTokens = tokenizer.encode(text: preparedPrompt)

        return (prompt, promptTokens)
    }

    func generate(
        promptTokens: [Int], model: LLMModel, tokenizer: Tokenizer,
        extraEOSTokens: Set<String>? = nil
    )
        -> GenerateResult
    {
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)

        return LLM.generate(
            promptTokens: promptTokens, parameters: generateParameters,
            model: model, tokenizer: tokenizer, extraEOSTokens: extraEOSTokens
        ) { tokens in

            if let last = tokens.last {
                detokenizer.append(token: last)
            }

            if let new = detokenizer.next() {
                print(new, terminator: "")
                fflush(stdout)
            }

            if tokens.count >= maxTokens {
                return .stop
            } else {
                return .more
            }
        }
    }
}

/// Argument package for adjusting and reporting memory use.
struct MemoryArguments: ParsableArguments, Sendable {

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

    static let configuration = CommandConfiguration(
        commandName: "eval",
        abstract: "evaluate prompt and generate text"
    )

    @OptionGroup var args: ModelArguments
    @OptionGroup var memory: MemoryArguments
    @OptionGroup var generate: GenerateArguments

    @MainActor
    mutating func run() async throws {
        let (modelContainer, modelConfiguration) = try await memory.start(args.load)

        if !generate.quiet {
            print("Model loaded -> \(modelConfiguration.id)")
        }

        let (prompt, promptTokens) = try await modelContainer.perform { [generate] _, tokenizer in
            try generate.tokenizePrompt(
                configuration: modelConfiguration, tokenizer: tokenizer)
        }

        if !generate.quiet {
            print("Starting generation ...")
            print(prompt, terminator: "")
        }

        let result = await modelContainer.perform { [generate] model, tokenizer in
            generate.generate(
                promptTokens: promptTokens, model: model, tokenizer: tokenizer,
                extraEOSTokens: modelConfiguration.extraEOSTokens)
        }
        print()

        if !generate.quiet {
            print("------")
            print(result.summary())

            memory.reportMemoryStatistics()
        }
    }
}
