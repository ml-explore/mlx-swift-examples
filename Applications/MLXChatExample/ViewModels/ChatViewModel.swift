//
//  ChatViewModel.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import Foundation
import LMResponseParserMLX
import MLXLMCommon
import UniformTypeIdentifiers

/// ViewModel that manages the chat interface and coordinates with MLXService for text generation.
/// Handles user input, message history, media attachments, and generation state.
@Observable
@MainActor
class ChatViewModel {
    /// Service responsible for ML model operations
    private let mlxService: MLXService

    init(mlxService: MLXService) {
        self.mlxService = mlxService
    }

    /// Current user input text
    var prompt: String = ""

    /// Chat history containing system, user, and assistant messages.
    ///
    /// `messages` is the on-screen transcript and the source of truth for what
    /// the model sees. ``MLXService`` reuses its active session as long as the
    /// selected model does not change; when it does, the session is rebuilt and
    /// seeded from this array so the new model continues the conversation.
    var messages: [Message] = [
        .system(MLXService.systemPrompt)
    ]

    /// Currently selected language model for generation
    var selectedModel: LMModel = MLXService.availableModels.first!

    /// Manages image and video attachments for the current message
    var mediaSelection = MediaSelection()

    /// Indicates if text generation is in progress
    var isGenerating = false

    /// Current generation task, used for cancellation
    private var generateTask: Task<Void, any Error>?

    /// Tokens per second for the most recently finalized turn, derived
    /// from `ResponseChatSession`'s output token count and the turn's
    /// wall-clock duration. `nil` until the first turn finalizes.
    private(set) var tokensPerSecond: Double?

    /// Whether there is any user-visible chat history that can be cleared.
    var canClearChat: Bool {
        messages.contains { $0.role != .system }
    }

    /// Progress of the current model download, if any
    var modelDownloadProgress: Progress? {
        mlxService.modelDownloadProgress
    }

    /// Whether a model is currently being initialized into memory.
    var isLoadingModel: Bool {
        mlxService.isLoadingModel
    }

    /// Most recent error message, if any
    var errorMessage: String?

    /// Generates response for the current prompt and media attachments
    func generate() async {
        if let existingTask = generateTask {
            existingTask.cancel()
            generateTask = nil
        }

        isGenerating = true

        // Capture the prompt/media before clearing the input.
        let userPrompt = prompt
        let userImages = mediaSelection.images
        let userVideos = mediaSelection.videos
        let model = selectedModel

        // Prior conversation excluding the system prompt (the session sets
        // it via `instructions`). Only consumed when the service rebuilds
        // its session after a model switch; otherwise ignored.
        let history: [Chat.Message] = messages.compactMap { message in
            switch message.role {
            case .user:
                return .user(
                    message.content,
                    images: message.images.map { .url($0) },
                    videos: message.videos.map { .url($0) }
                )
            case .assistant:
                return .assistant(message.content)
            case .system:
                return nil
            }
        }

        messages.append(.user(userPrompt, images: userImages, videos: userVideos))
        messages.append(.assistant(""))

        clear(.prompt)

        generateTask = Task {
            let stream = try await mlxService.generate(
                history: history,
                prompt: userPrompt,
                images: userImages,
                videos: userVideos,
                model: model
            )
            let startTime = Date.now
            for try await event in stream {
                guard let assistantMessage = messages.last else { continue }
                switch event {
                case .outputTextDelta(let delta):
                    appendText(.content, itemId: delta.itemId, delta: delta.delta,
                               to: assistantMessage)

                case .reasoningTextDelta(let delta):
                    appendText(.reasoning, itemId: delta.itemId, delta: delta.delta,
                               to: assistantMessage)

                case .outputItemAdded(let added):
                    switch added.item {
                    case .functionCall(let call):
                        assistantMessage.segments.append(
                            .toolCall(
                                ToolCall(
                                    id: call.id,
                                    callId: call.callId,
                                    name: call.name,
                                    argumentsRaw: call.arguments
                                )
                            )
                        )
                    case .functionCallOutput(let output):
                        if let toolCall = findToolCall(callId: output.callId,
                                                       in: assistantMessage) {
                            toolCall.result = output.output.stringValue ?? ""
                        }
                    case .message, .reasoning:
                        break
                    }

                case .functionCallArgumentsDelta(let delta):
                    if let toolCall = findToolCall(itemId: delta.itemId,
                                                   in: assistantMessage) {
                        toolCall.argumentsRaw += delta.delta
                    }

                default:
                    break
                }
            }
            if let outputTokens = mlxService.lastResponseOutputTokens {
                let elapsed = Date.now.timeIntervalSince(startTime)
                if elapsed > 0 {
                    tokensPerSecond = Double(outputTokens) / elapsed
                }
            }
        }

        do {
            try await withTaskCancellationHandler {
                try await generateTask?.value
            } onCancel: {
                Task { @MainActor in
                    generateTask?.cancel()

                    if let assistantMessage = messages.last {
                        assistantMessage.segments.append(
                            .content(TextSegment(itemId: "_cancelled", text: "\n[Cancelled]"))
                        )
                    }

                    // Drop the session so the next turn rebuilds from
                    // the visible history. The KV cache holds a partial
                    // assistant response with no end-of-turn marker;
                    // reusing it would feed the next call a malformed
                    // transcript.
                    mlxService.clearSession()
                }
            }
        } catch is CancellationError {
            // Stop, dismiss, or chat-clear mid-generation – not surfaced.
        } catch let error as URLError where error.code == .cancelled {
            // Same intent, raised from an in-flight URLSession (e.g.
            // model download interrupted).
        } catch {
            errorMessage = error.localizedDescription
        }

        isGenerating = false
        generateTask = nil
    }

    private enum TextSegmentKind { case content, reasoning }

    /// Append a delta to the latest segment if it matches kind and
    /// itemId; otherwise open a new segment. Drives the interleaving of
    /// reasoning, tool calls, and content in arrival order.
    private func appendText(
        _ kind: TextSegmentKind,
        itemId: String,
        delta: String,
        to message: Message
    ) {
        if case let .reasoning(segment) = message.segments.last,
           kind == .reasoning, segment.itemId == itemId
        {
            segment.text += delta
            return
        }
        if case let .content(segment) = message.segments.last,
           kind == .content, segment.itemId == itemId
        {
            segment.text += delta
            return
        }
        let new = TextSegment(itemId: itemId, text: delta)
        switch kind {
        case .content:
            message.segments.append(.content(new))
        case .reasoning:
            message.segments.append(.reasoning(new))
        }
    }

    private func findToolCall(itemId: String, in message: Message) -> ToolCall? {
        for segment in message.segments {
            if case let .toolCall(call) = segment, call.id == itemId { return call }
        }
        return nil
    }

    private func findToolCall(callId: String, in message: Message) -> ToolCall? {
        for segment in message.segments {
            if case let .toolCall(call) = segment, call.callId == callId { return call }
        }
        return nil
    }

    /// Processes and adds media attachments to the current message
    func addMedia(_ result: Result<URL, any Error>) {
        do {
            let url = try result.get()

            // Determine media type and add to appropriate collection
            if let mediaType = UTType(filenameExtension: url.pathExtension) {
                if mediaType.conforms(to: .image) {
                    mediaSelection.images = [url]
                } else if mediaType.conforms(to: .movie) {
                    mediaSelection.videos = [url]
                }
            }
        } catch {
            errorMessage = "Failed to load media item.\n\nError: \(error)"
        }
    }

    /// Clears various aspects of the chat state based on provided options
    func clear(_ options: ClearOption) {
        if options.contains(.prompt) {
            prompt = ""
            mediaSelection = .init()
        }

        if options.contains(.chat) {
            generateTask?.cancel()
            messages = [.system(MLXService.systemPrompt)]
            mlxService.clearSession()
        }

        if options.contains(.meta) {
            tokensPerSecond = nil
        }

        errorMessage = nil
    }
}

/// Manages the state of media attachments in the chat
@Observable
class MediaSelection {
    /// Controls visibility of media selection UI
    var isShowing = false

    /// Currently selected image URLs
    var images: [URL] = [] {
        didSet {
            didSetURLs(oldValue, images)
        }
    }

    /// Currently selected video URLs
    var videos: [URL] = [] {
        didSet {
            didSetURLs(oldValue, videos)
        }
    }

    /// Whether any media is currently selected
    var isEmpty: Bool {
        images.isEmpty && videos.isEmpty
    }

    private func didSetURLs(_ old: [URL], _ new: [URL]) {
        // the urls we get from fileImporter require SSB calls to access
        new.filter { !old.contains($0) }.forEach { _ = $0.startAccessingSecurityScopedResource() }
        old.filter { !new.contains($0) }.forEach { $0.stopAccessingSecurityScopedResource() }
    }
}

/// Options for clearing different aspects of the chat state
struct ClearOption: RawRepresentable, OptionSet {
    let rawValue: Int

    /// Clears current prompt and media selection
    static let prompt = ClearOption(rawValue: 1 << 0)
    /// Clears chat history and cancels generation
    static let chat = ClearOption(rawValue: 1 << 1)
    /// Clears generation metadata
    static let meta = ClearOption(rawValue: 1 << 2)
}
