// Copyright © 2025 Apple Inc.

import MLXLLM
import MLXLMCommon
import SwiftUI

/// which model to load
private let modelConfiguration = LLMRegistry.gemma3_1B_qat_4bit

/// instructions for the model (the system prompt)
private let instructions =
    """
    You are a friendly and helpful chatbot.
    """

/// parameters controlling generation
private let generateParameters = GenerateParameters(temperature: 0.5)

/// Downloads and loads the weights for the model -- we have one of these in the process
@MainActor @Observable public class ModelLoader {

    enum State {
        case idle
        case loading(Task<ModelContainer, Error>)
        case loaded(ModelContainer)
    }

    public var progress = 0.0
    public var isLoaded: Bool {
        switch state {
        case .idle, .loading: false
        case .loaded: true
        }
    }

    private var state = State.idle

    public func model() async throws -> ModelContainer {
        switch self.state {
        case .idle:
            let task = Task {
                // download and report progress
                try await loadModelContainer(configuration: modelConfiguration) { value in
                    Task { @MainActor in
                        self.progress = value.fractionCompleted
                    }
                }
            }
            self.state = .loading(task)
            let model = try await task.value

            self.state = .loaded(model)
            return model

        case .loading(let task):
            return try await task.value

        case .loaded(let model):
            return model
        }
    }
}

/// View model for the ChatSession
@MainActor @Observable public class ChatModel {

    private let session: ChatSession

    /// back and forth conversation between the user and LLM
    public var messages = [Chat.Message]()

    private var task: Task<Void, Error>?
    public var isBusy: Bool {
        task != nil
    }

    public init(model: ModelContainer) {
        self.session = ChatSession(
            model,
            instructions: instructions,
            generateParameters: generateParameters)
    }

    public func cancel() {
        task?.cancel()
    }

    public func respond(_ message: String) {
        guard task == nil else { return }

        self.messages.append(.init(role: .user, content: message))
        self.messages.append(.init(role: .assistant, content: "..."))
        let lastIndex = self.messages.count - 1

        self.task = Task {
            var first = true
            for try await item in session.streamResponse(to: message) {
                if first {
                    self.messages[lastIndex].content = item
                    first = false
                } else {
                    self.messages[lastIndex].content += item
                }
            }
            self.task = nil
        }
    }
}
