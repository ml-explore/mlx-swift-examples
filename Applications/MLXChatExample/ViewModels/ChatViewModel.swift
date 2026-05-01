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

    /// Current generation speed in tokens per second, if a generation has completed.
    /// Computed from `ResponseChatSession`'s reported output token count and the
    /// wall-clock time the turn took. `nil` until the first turn finalizes.
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

        // Capture the current prompt/media so the input can be cleared before the
        // task starts.
        let userPrompt = prompt
        let userImages = mediaSelection.images
        let userVideos = mediaSelection.videos
        let model = selectedModel

        // Build the prior conversation (excluding the system prompt, which the
        // session sets via `instructions`) so `MLXService` can seed a fresh
        // session if the user just switched models. For follow-up turns on the
        // same model the service ignores this and reuses its existing session.
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
                switch event {
                case .outputTextDelta(let delta):
                    if let assistantMessage = messages.last {
                        assistantMessage.content += delta.delta
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
                        assistantMessage.content += "\n[Cancelled]"
                    }

                    // Drop the active session so the next turn rebuilds from the
                    // visible history. The session's KV cache currently holds a
                    // partial assistant response with no end-of-turn marker;
                    // continuing on top of it would feed the next model call a
                    // malformed transcript.
                    mlxService.clearSession()
                }
            }
        } catch is CancellationError {
            // Expected when the user stops generation, dismisses the view, or
            // clears the chat mid-generation – not surfaced to the user.
        } catch let error as URLError where error.code == .cancelled {
            // Same intent as above, when the cancellation reaches an in-flight
            // URLSession task (e.g. a model download).
        } catch {
            errorMessage = error.localizedDescription
        }

        isGenerating = false
        generateTask = nil
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
