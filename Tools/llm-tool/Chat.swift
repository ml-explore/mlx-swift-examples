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

    mutating func run() async throws {
        let defaultModel = MLXLLM.LLMRegistry.mistral7B4bit

        // Load the model
        let modelContainer = try await memory.start { [args] in
            do {
                return try await args.load(
                    defaultModel: defaultModel.name, modelFactory: VLMModelFactory.shared)
            } catch ModelFactoryError.unsupportedModelType {
                return try await args.load(
                    defaultModel: defaultModel.name, modelFactory: LLMModelFactory.shared)
            }
        }

        // update the context/configuration with any command line parameters
        await modelContainer.update { [generate] context in
            generate.prepare(&context)
        }

        try await chat(modelContainer: modelContainer)
    }

    func chat(modelContainer: ModelContainer) async throws {
        let session = ChatSession(
            modelContainer,
            instructions: generate.system,
            generateParameters: generate.generateParameters,
            processing: media.processing
        )

        var printStats = false
        var images: [UserInput.Image] = []
        var videos: [UserInput.Video] = []

        while true {
            print("\n\n> ", terminator: "")
            guard let line = readLine() else {
                return
            }

            if line.hasPrefix("/") {
                let command = line.split(separator: " ")[0]
                let rest = String(
                    line.dropFirst(command.count).trimmingCharacters(in: .whitespaces))

                func url(_ string: String) -> URL? {
                    if string.hasPrefix("/") || !string.hasPrefix("http") {
                        URL(filePath: string)
                    } else {
                        URL(string: string)
                    }
                }

                switch command {
                case "/help":
                    help()

                case "/quit":
                    return

                case "/memory":
                    let memory = Memory.snapshot()
                    print("Memory size: \(Memory.memoryLimit / 1024)K")
                    print("Cache size:  \(Memory.cacheLimit / 1024)K")
                    print(memory.description)

                case "/stats":
                    printStats.toggle()
                    print("Token stats: \(printStats ? "ON" : "OFF")")

                case "/reset":
                    await session.clear()

                case "/image":
                    if let url = url(rest) {
                        images.append(.url(url))
                    }
                case "/video":
                    if let url = url(rest) {
                        videos.append(.url(url))
                    }

                case "/parameters":
                    print(session.generateParameters)
                case "/temperature":
                    if let value = Float(rest) {
                        session.generateParameters.temperature = value
                        print(session.generateParameters)
                    }
                case "/topP":
                    if let value = Float(rest) {
                        session.generateParameters.topP = value
                        print(session.generateParameters)
                    }
                case "/maxTokens":
                    session.generateParameters.maxTokens = Int(rest)
                    print(session.generateParameters)

                default:
                    help()
                }
                continue

            } else if line.isEmpty {
                continue
            }

            // generate the output

            var result: GenerateCompletionInfo?
            for try await item in session.streamDetails(
                to: line, images: images, videos: videos
            ) {
                switch item {
                case .chunk(let string):
                    print(string, terminator: "")
                case .info(let info):
                    result = info
                case .toolCall:
                    break
                }
            }

            // these have been presented, remove
            images.removeAll()
            videos.removeAll()

            if printStats, let result {
                print(
                    "\ntime to first token: \(result.promptTime.formatted()) tps: \(result.tokensPerSecond.formatted())"
                )
            }
        }
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
            /parameters -- print generation parametes
            /temperature [number] -- set the sampling temperature
            /topP [number] -- set the top p sampling
            /maxTokens [number] -- set the maximum number of tokens to generate or no number to remove limit
            """)
    }
}
