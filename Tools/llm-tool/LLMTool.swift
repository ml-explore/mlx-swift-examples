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
        subcommands: [EvaluateCommand.self, ChatCommand.self, LoRACommand.self],
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

struct PromptArguments: ParsableArguments, Sendable {
    @Option(
        name: .shortAndLong,
        help:
            "The message to be processed by the model.  Use @path,@path to load from files, e.g. @/tmp/prompt.txt"
    )
    var prompt: String?

    func resolvePrompt(configuration: ModelConfiguration) throws -> String {
        let prompt = self.prompt ?? configuration.defaultPrompt
        if prompt.hasPrefix("@") {
            let names = prompt.split(separator: ",").map { String($0.dropFirst()) }
            return try names.map { try String(contentsOfFile: $0) }.joined(separator: "\n")
        } else {
            return prompt
        }
    }
}

/// Argument package for supplying media files
struct MediaArguments: ParsableArguments, Sendable {

    @Option(parsing: .upToNextOption, help: "Resize images to this size (width, height)")
    var resize: [Int] = []

    @Option(parsing: .upToNextOption, help: "Paths or URLs for input images")
    var image: [URL] = []

    @Option(parsing: .upToNextOption, help: "Paths or URLs for input videos")
    var video: [URL] = []

    var images: [UserInput.Image] {
        image.map { UserInput.Image.url($0) }
    }
    var videos: [UserInput.Video] {
        video.map { UserInput.Video.url($0) }
    }

    var processing: UserInput.Processing {
        var processing = UserInput.Processing()
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
            processing.resize = size
        }
        return processing
    }
}

/// Command line arguments for controlling generation of text.
struct GenerateArguments: ParsableArguments, Sendable {

    @Option(
        name: .shortAndLong,
        help:
            "The system prompt"
    )
    var system: String = ""

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
            maxTokens: maxTokens,
            temperature: temperature, topP: topP, repetitionPenalty: repetitionPenalty,
            repetitionContextSize: repetitionContextSize)
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
    ) async throws -> (GenerateCompletionInfo, String) {
        var output = ""
        for await item in try MLXLMCommon.generate(
            input: input, parameters: generateParameters, context: context)
        {
            switch item {
            case .chunk(let string):
                output += string
                print(string, terminator: "")
            case .info(let info):
                return (info, output)
            }
        }
        fatalError("exited loop without seeing .info")
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
    @OptionGroup var prompt: PromptArguments
    @OptionGroup var media: MediaArguments

    private func userInput(modelConfiguration: ModelConfiguration) -> UserInput {
        let prompt =
            (try? self.prompt.resolvePrompt(configuration: modelConfiguration))
            ?? modelConfiguration.defaultPrompt

        return UserInput(
            chat: [
                .system(generate.system),
                .user(prompt, images: media.images, videos: media.videos),
            ],
            processing: media.processing
        )
    }

    @MainActor
    mutating func run() async throws {
        let modelFactory: ModelFactory
        let defaultModel: ModelConfiguration

        // Switch between LLM and VLM based on presence of media
        let vlm = !media.image.isEmpty || !media.video.isEmpty
        if vlm {
            modelFactory = VLMModelFactory.shared
            defaultModel = MLXVLM.VLMRegistry.qwen2VL2BInstruct4Bit
        } else {
            modelFactory = LLMModelFactory.shared
            defaultModel = MLXLLM.LLMRegistry.mistral7B4bit
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

        let (result, _) = try await modelContainer.perform { [generate] context in
            let input = try await context.processor.prepare(input: userInput)
            return try await generate.generate(input: input, context: context)
        }

        if !generate.quiet {
            print("------")
            print(result.summary())

            memory.reportMemoryStatistics()
        }
    }
}
