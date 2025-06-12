// Copyright © 2025 Apple Inc.

import Foundation
import MLX

private class Generator {

    enum Model {
        case container(ModelContainer)
        case context(ModelContext)
    }

    let model: Model
    var messages = [Chat.Message]()
    let processing: UserInput.Processing
    let generateParameters: GenerateParameters
    var cache: [KVCache]

    init(
        model: Model, instructions: String?, prompt: String, image: UserInput.Image?,
        video: UserInput.Video?, processing: UserInput.Processing,
        generateParameters: GenerateParameters
    ) {
        self.model = model
        self.messages = []
        if let instructions = instructions {
            messages.append(.system(instructions))
        }
        messages.append(
            .user(
                prompt, images: image.flatMap { [$0] } ?? [], videos: video.flatMap { [$0] } ?? []))
        self.processing = processing
        self.generateParameters = generateParameters
        self.cache = []
    }

    init(
        model: Model, instructions: String?, processing: UserInput.Processing,
        generateParameters: GenerateParameters
    ) {
        self.model = model
        if let instructions {
            self.messages = [.system(instructions)]
        } else {
            self.messages = []
        }
        self.processing = processing
        self.generateParameters = generateParameters
        self.cache = []
    }

    func generate() async throws -> String {
        func generate(context: ModelContext) async throws -> String {
            let userInput = UserInput(chat: messages, processing: processing)
            let input = try await context.processor.prepare(input: userInput)

            if cache.isEmpty {
                cache = context.model.newCache(parameters: generateParameters)
            }
            let iterator = try TokenIterator(
                input: input, model: context.model, cache: cache, parameters: generateParameters)
            let result: GenerateResult = MLXLMCommon.generate(
                input: input, context: context, iterator: iterator
            ) { _ in .more }

            Stream.gpu.synchronize()

            return result.output
        }

        switch model {
        case .container(let container):
            return try await container.perform { context in
                try await generate(context: context)
            }
        case .context(let context):
            return try await generate(context: context)
        }
    }

    func stream() -> AsyncThrowingStream<String, Error> {
        func stream(
            context: ModelContext,
            continuation: AsyncThrowingStream<String, Error>.Continuation
        ) async {
            do {
                let userInput = UserInput(chat: messages, processing: processing)
                let input = try await context.processor.prepare(input: userInput)

                if cache.isEmpty {
                    cache = context.model.newCache(parameters: generateParameters)
                }

                for await item in try MLXLMCommon.generate(
                    input: input, cache: cache, parameters: generateParameters, context: context)
                {
                    switch item {
                    case .chunk(let chunk): continuation.yield(chunk)
                    case .info: break
                    }
                }

                Stream.gpu.synchronize()

                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        return AsyncThrowingStream { continuation in
            Task { [model, continuation] in
                switch model {
                case .container(let container):
                    await container.perform { context in
                        await stream(context: context, continuation: continuation)
                    }
                case .context(let context):
                    await stream(context: context, continuation: continuation)
                }
            }
        }
    }
}

public class ChatSession {

    private let generator: Generator

    public init(
        _ model: ModelContainer, instructions: String? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512))
    ) {
        self.generator = .init(
            model: .container(model), instructions: instructions, processing: processing,
            generateParameters: generateParameters)
    }

    public init(
        _ model: ModelContext, instructions: String? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512))
    ) {
        self.generator = .init(
            model: .context(model), instructions: instructions, processing: processing,
            generateParameters: generateParameters)
    }

    public func respond(
        to prompt: String, image: UserInput.Image? = nil, video: UserInput.Video? = nil
    ) async throws -> String {
        generator.messages = [
            .user(
                prompt,
                images: image.flatMap { [$0] } ?? [],
                videos: video.flatMap { [$0] } ?? [])
        ]
        return try await generator.generate()
    }

    public func streamResponse(
        to prompt: String, image: UserInput.Image? = nil, video: UserInput.Video? = nil
    ) -> AsyncThrowingStream<String, Error> {
        generator.messages = [
            .user(
                prompt,
                images: image.flatMap { [$0] } ?? [],
                videos: video.flatMap { [$0] } ?? [])
        ]
        return generator.stream()
    }

}
