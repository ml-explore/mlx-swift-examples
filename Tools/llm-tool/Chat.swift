// Copyright © 2025 Apple Inc.

import ArgumentParser
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

struct ChatCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "chat",
        abstract: "interactive chat with model"
    )

    @OptionGroup var args: ModelArguments
    @OptionGroup var memory: MemoryArguments
    @OptionGroup var generate: GenerateArguments
    @OptionGroup var media: MediaArguments

    struct State {
        var parameters: GenerateParameters
        var processing: UserInput.Processing

        var images: [UserInput.Image]
        var videos: [UserInput.Video]

        var chat: [Chat.Message]

        var cache: [KVCache]

        var printStats = false
    }

    @MainActor
    mutating func run() async throws {
        let defaultModel = MLXLLM.LLMRegistry.mistral7B4bit

        // Load the model
        let modelContainer = try await memory.start { [args] in
            do {
                return try await args.load(
                    defaultModel: defaultModel.name, modelFactory: LLMModelFactory.shared)
            } catch ModelFactoryError.unsupportedModelType {
                return try await args.load(
                    defaultModel: defaultModel.name, modelFactory: VLMModelFactory.shared)
            }
        }

        // update the context/configuration with any command line parameters
        await modelContainer.update { [generate] context in
            generate.prepare(&context)
        }

        try await chat(modelContainer: modelContainer)
    }

    func chat(modelContainer: ModelContainer) async throws {
        try await modelContainer.perform { context in
            let parameters = generate.generateParameters
            let initialState = State(
                parameters: parameters,
                processing: media.processing,
                images: media.images, videos: media.videos,
                chat: [.system(generate.system)],
                cache: context.model.newCache(parameters: parameters))

            var state = initialState

            print("> ", terminator: "")
            while let line = readLine() {
                if line.hasPrefix("/") {
                    // handle commands
                    switch command(line: line, state: &state) {
                    case .exit:
                        return
                    case .reset:
                        state = initialState
                        state.cache = context.model.newCache(parameters: parameters)
                        continue
                    case .inference:
                        // continue and run inference
                        break
                    case .handled:
                        print("\n\n> ", terminator: "")
                        continue
                    }
                } else {
                    // chat input
                    state.chat.append(.user(line, images: state.images, videos: state.videos))
                }

                // consume the media, if any
                state.images.removeAll()
                state.videos.removeAll()

                // convert UserInput to LMInput
                let userInput = UserInput(chat: state.chat, processing: state.processing)
                let input = try await context.processor.prepare(input: userInput)

                // generate the output
                var output = ""
                var result: GenerateCompletionInfo?
                for await item in try MLXLMCommon.generate(
                    input: input, cache: state.cache, parameters: parameters, context: context
                ) {
                    switch item {
                    case .chunk(let string):
                        output += string
                        print(string, terminator: "")
                    case .info(let info):
                        result = info
                    }
                }

                // add the assistant response to the chat messages
                state.chat.append(.assistant(output))

                if state.printStats, let result {
                    print(
                        "\ntime to first token: \(result.promptTime.formatted()) tps: \(result.tokensPerSecond.formatted())"
                    )
                }
                print("\n\n> ", terminator: "")
            }
        }
    }

    enum CommandDisposition {
        case exit
        case reset
        case inference
        case handled
    }

    func help() {
        print(
            """
            /help -- this message
            /quit -- terminate the chat
            /memory -- print memory stats
            /stats -- toggle token stats
            /reset -- reset the chat session to initial state
            /image [pathOrURL] -- provide an image
            /video [pathOrURL] -- provide a video
            /again -- rerun inference for last response
            /parameters -- print generation parametes
            /temperature [number] -- set the sampling temperature
            /topP [number] -- set the top p sampling
            /maxTokens [number] -- set the maximum number of tokens to generate or no number to remove limit
            """)
    }

    func command(line: String, state: inout State) -> CommandDisposition {
        let command = line.split(separator: " ")[0]
        let rest = String(
            line.dropFirst(command.count).trimmingCharacters(in: .whitespaces))

        func url(_ string: String) -> URL? {
            if string.hasPrefix("/") {
                URL(filePath: string)
            } else {
                URL(string: string)
            }
        }

        switch command {
        case "/help":
            help()

        case "/quit":
            return .exit

        case "/memory":
            let memory = GPU.snapshot()
            print("Memory size: \(GPU.memoryLimit / 1024)K")
            print("Cache size:  \(GPU.cacheLimit / 1024)K")
            print(memory.description)

        case "/stats":
            state.printStats.toggle()
            print("Token stats: \(state.printStats ? "ON" : "OFF")")

        case "/reset":
            return .reset

        case "/image":
            if let url = url(rest) {
                state.images.append(UserInput.Image.url(url))
            }
        case "/video":
            if let url = url(rest) {
                state.videos.append(UserInput.Video.url(url))
            }

        case "/again":
            state.chat.removeLast()
            return .inference

        case "/parameters":
            print(state.parameters)
        case "/temperature":
            if let value = Float(rest) {
                state.parameters.temperature = value
                print(state.parameters)
            }
        case "/topP":
            if let value = Float(rest) {
                state.parameters.topP = value
                print(state.parameters)
            }
        case "/maxTokens":
            state.parameters.maxTokens = Int(rest)
            print(state.parameters)

        default:
            help()
        }

        return .handled
    }
}
