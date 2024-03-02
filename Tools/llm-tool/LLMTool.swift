// Copyright © 2024 Apple Inc.

import ArgumentParser
import Foundation
import LLM
import MLX
import MLXRandom

@main
struct LLMTool: AsyncParsableCommand {
    static var configuration = CommandConfiguration(
        abstract: "Command line tool for generating text using Llama models",
        subcommands: [SyncGenerator.self, AsyncGenerator.self],
        defaultSubcommand: SyncGenerator.self)
}

struct SyncGenerator: AsyncParsableCommand {

    static var configuration = CommandConfiguration(
        commandName: "sync",
        abstract: "Synchronous generator"
    )

    @Option(name: .long, help: "Name of the huggingface model")
    var model: String = "mlx-community/Mistral-7B-v0.1-hf-4bit-mlx"

    @Option(name: .shortAndLong, help: "The message to be processed by the model")
    var prompt = "compare python and swift"

    @Option(name: .shortAndLong, help: "Maximum number of tokens to generate")
    var maxTokens = 100

    @Option(name: .shortAndLong, help: "The sampling temperature")
    var temperature: Float = 0.6

    @Option(name: .long, help: "The PRNG seed")
    var seed: UInt64 = 0

    @MainActor
    func run() async throws {
        MLXRandom.seed(seed)

        let modelConfiguration = ModelConfiguration.configuration(id: model)
        let (model, tokenizer) = try await load(configuration: modelConfiguration)

        let prompt = modelConfiguration.prepare(prompt: self.prompt)
        let promptTokens = tokenizer.encode(text: prompt)

        print("Starting generation ...")
        print(self.prompt, terminator: "")

        var start = Date.timeIntervalSinceReferenceDate
        var promptTime: TimeInterval = 0

        // collect the tokens and keep track of how much of the string
        // we have printed already
        var tokens = [Int]()
        var printed = 0

        for token in TokenIterator(prompt: MLXArray(promptTokens), model: model, temp: temperature)
        {
            if tokens.isEmpty {
                eval(token)
                let now = Date.timeIntervalSinceReferenceDate
                promptTime = now - start
                start = now
            }

            let t = token.item(Int.self)
            if t == tokenizer.unknownTokenId || t == tokenizer.eosTokenId {
                break
            }
            tokens.append(t)

            // print any new parts of the string
            let fullOutput = tokenizer.decode(tokens: tokens)
            let emitLength = fullOutput.count - printed
            let suffix = fullOutput.suffix(emitLength)
            print(suffix, terminator: "")
            fflush(stdout)

            printed = fullOutput.count

            if tokens.count == maxTokens {
                break
            }
        }

        print()
        print("------")
        let now = Date.timeIntervalSinceReferenceDate
        let generateTime = now - start

        print(
            """
            Prompt Tokens per second:     \((Double(promptTokens.count) / promptTime).formatted())
            Generation tokens per second: \((Double(tokens.count - 1) / generateTime).formatted())
            """)
    }
}

/// Example of an async generator.
///
/// Note that all of the computation is done on another thread and TokenId (Int32) are sent
/// rather than MLXArray.
struct AsyncGenerator: AsyncParsableCommand {

    static var configuration = CommandConfiguration(
        commandName: "async",
        abstract: "async generator"
    )

    @Option(name: .long, help: "Name of the huggingface model")
    var model: String = "mlx-community/Mistral-7B-v0.1-hf-4bit-mlx"

    @Option(name: .shortAndLong, help: "The message to be processed by the model")
    var prompt = "compare python and swift"

    @Option(name: .shortAndLong, help: "Maximum number of tokens to generate")
    var maxTokens = 100

    @Option(name: .shortAndLong, help: "The sampling temperature")
    var temperature: Float = 0.6

    @Option(name: .long, help: "The PRNG seed")
    var seed: UInt64 = 0

    @MainActor
    func run() async throws {
        MLXRandom.seed(seed)

        let modelConfiguration = ModelConfiguration.configuration(id: model)
        let (model, tokenizer) = try await load(configuration: modelConfiguration)

        let prompt = modelConfiguration.prepare(prompt: self.prompt)
        let promptTokens = tokenizer.encode(text: prompt)

        print("Starting generation ...")
        print(self.prompt, terminator: "")

        var start = Date.timeIntervalSinceReferenceDate
        var promptTime: TimeInterval = 0

        // collect the tokens and keep track of how much of the string
        // we have printed already
        var tokens = [Int]()
        var printed = 0

        let (task, channel) = generate(
            prompt: MLXArray(promptTokens), model: model, temp: temperature)

        for await token in channel {
            if tokens.isEmpty {
                let now = Date.timeIntervalSinceReferenceDate
                promptTime = now - start
                start = now
            }

            if token == tokenizer.unknownTokenId || token == tokenizer.eosTokenId {
                break
            }
            tokens.append(token)

            // print any new parts of the string
            let fullOutput = tokenizer.decode(tokens: tokens)
            let emitLength = fullOutput.count - printed
            let suffix = fullOutput.suffix(emitLength)
            print(suffix, terminator: "")
            fflush(stdout)

            printed = fullOutput.count

            if tokens.count == maxTokens {
                break
            }
        }

        // tell the task to stop
        task.cancel()

        print()
        print("------")
        let now = Date.timeIntervalSinceReferenceDate
        let generateTime = now - start

        print(
            """
            Prompt Tokens per second:     \((Double(promptTokens.count) / promptTime).formatted())
            Generation tokens per second: \((Double(tokens.count - 1) / generateTime).formatted())
            """)

        // wait for the task to complete -- since it is running async, it might
        // be in the middle of running the model
        try? await Task.sleep(for: .milliseconds(500))
    }
}
