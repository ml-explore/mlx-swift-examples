// Copyright © 2024 Apple Inc.

import ArgumentParser
import CoreImage
import Foundation
import Hub
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import MLXVLM
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

    @Option(name: .long, help: "Name of the Hugging Face model or absolute path to directory")
    var model: String?

    @Sendable
    func load(defaultModel: String, modelFactory: ModelFactory) async throws -> ModelContainer {
        let modelConfiguration: ModelConfiguration

        let modelName = self.model ?? defaultModel

        print("Loading \(modelName)...")

        if modelName.hasPrefix("/") {
            // path
            modelConfiguration = ModelConfiguration(directory: URL(filePath: modelName))
        } else {
            // identifier
            modelConfiguration = modelFactory.configuration(id: modelName)
        }
        return try await modelFactory.loadContainer(configuration: modelConfiguration)
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

    @Option(name: .long, help: "Additional end-of-sequence token to stop generation")
    var extraEosToken: String?

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

    func prepare(
        _ context: inout ModelContext
    ) {
        if let extraEosToken {
            context.configuration.extraEOSTokens.insert(extraEosToken)
        }
    }

    func generate(
        input: LMInput, context: ModelContext
    ) throws -> GenerateResult {
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)

        return try MLXLMCommon.generate(
            input: input, parameters: generateParameters, context: context
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

    mutating func start<L>(_ load: @Sendable () async throws -> L) async throws -> L {
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

    @Option(parsing: .upToNextOption, help: "Resize images to this size (width, height)")
    var resize: [Int] = []

    @Option(parsing: .upToNextOption, help: "Paths or URLs for input images")
    var image: [URL] = []

    @Option(parsing: .upToNextOption, help: "Paths or URLs for input videos")
    var video: [URL] = []

    private func userInput(modelConfiguration: ModelConfiguration) -> UserInput {
        let prompt =
            (try? generate.resolvePrompt(configuration: modelConfiguration))
            ?? modelConfiguration.defaultPrompt
        let images = image.map { UserInput.Image.url($0) }
        let videos = video.map { UserInput.Video.url($0) }
        let messages: [[String: Any]] =
            if !images.isEmpty || !videos.isEmpty {
                [
                    [
                        "role": "user",
                        "content": [
                            ["type": "text", "text": prompt]
                        ]
                            // Messages format for Qwen 2 VL, Qwen 2.5 VL. May need to be adapted for other models.
                            + images.map { _ in ["type": "image"] }
                            + videos.map { _ in ["type": "video"] },
                    ]
                ]
            } else {
                [
                    [
                        "role": "user",
                        "content": prompt,
                    ]
                ]
            }
        var userInput = UserInput(messages: messages, images: images, videos: videos)
        if !resize.isEmpty {
            let size: CGSize
            if resize.count == 1 {
                // Single value represents width/height
                let v = resize[0]
                size = CGSize(width: v, height: v)
            } else {
                let v0 = resize[0]
                let v1 = resize[1]
                size = CGSize(width: v0, height: v1)
            }
            userInput.processing.resize = size
        }
        return userInput
    }

    @MainActor
    mutating func run() async throws {
        let modelFactory: ModelFactory
        let defaultModel: ModelConfiguration

        // Switch between LLM and VLM based on presence of media
        let vlm = !image.isEmpty || !video.isEmpty
        if vlm {
            modelFactory = VLMModelFactory.shared
            defaultModel = MLXVLM.ModelRegistry.qwen2VL2BInstruct4Bit
        } else {
            modelFactory = LLMModelFactory.shared
            defaultModel = MLXLLM.ModelRegistry.mistral7B4bit
        }

        // Load the model
        let modelContainer = try await memory.start { [args] in
            try await args.load(defaultModel: defaultModel.name, modelFactory: modelFactory)
        }

        // update the context/configuration with any command line parameters
        await modelContainer.update { [generate] context in
            generate.prepare(&context)
        }

        // Get the resolved configuration (this has the default prompt)
        let modelConfiguration = await modelContainer.configuration

        if !generate.quiet {
            print("Loaded \(modelConfiguration.name)")
        }

        let userInput = self.userInput(modelConfiguration: modelConfiguration)

        if !generate.quiet {
            print("Starting generation ...")
            print(userInput.prompt, terminator: " ")
        }

        let result = try await modelContainer.perform { [generate] context in
            let input = try await context.processor.prepare(input: userInput)
            return try generate.generate(input: input, context: context)
        }

        if !generate.quiet {
            print("------")
            print(result.summary())

            memory.reportMemoryStatistics()
        }
    }
}
