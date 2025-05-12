//
//  ChatViewModel.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import Foundation
import MLXLMCommon
import UniformTypeIdentifiers

/// ViewModel that manages the chat interface and coordinates with MLXService for text generation.
/// Handles user input, message history, media attachments, and generation state.
@Observable
@MainActor
class ChatViewModel {
    /// Service responsible for ML model operations
    private let mlxService: MLXService

    /// Assistant display mode
    var displayMode: MessageDisplayMode = .markdown

    init(mlxService: MLXService) {
        self.mlxService = mlxService
    }

    /// Current user input text
    var prompt: String = ""

    /// Chat history containing system, user, and assistant messages
    var messages: [Message] = [
        .system("You are a helpful assistant!")
    ]

    /// Currently selected language model for generation
    var selectedModel: LMModel = MLXService.availableModels.first!

    /// Manages image and video attachments for the current message
    var mediaSelection = MediaSelection()

    /// Indicates if text generation is in progress
    var isGenerating = false

    /// Current generation task, used for cancellation
    private var generateTask: Task<Void, any Error>?

    /// Stores performance metrics from the current generation
    private var generateCompletionInfo: GenerateCompletionInfo?

    /// Current generation speed in tokens per second
    var tokensPerSecond: Double {
        generateCompletionInfo?.tokensPerSecond ?? 0
    }

    /// Progress of the current model download, if any
    var modelDownloadProgress: Progress? {
        mlxService.modelDownloadProgress
    }

    /// Most recent error message, if any
    var errorMessage: String?

    /// Generates response for the current prompt and media attachments
    func generate() async {
        // Cancel any existing generation task
        if let existingTask = generateTask {
            existingTask.cancel()
            generateTask = nil
        }

        isGenerating = true

        // Add user message with any media attachments
        messages.append(.user(prompt, images: mediaSelection.images, videos: mediaSelection.videos))
        // Add empty assistant message that will be filled during generation
        messages.append(.assistant(""))

        // Clear the input after sending
        clear(.prompt)

        generateTask = Task {
            // Process generation chunks and update UI
            for await generation in try await mlxService.generate(
                messages: messages, model: selectedModel)
            {
                switch generation {
                case .chunk(let chunk):
                    // Append new text to the current assistant message
                    if let assistantMessage = messages.last {
                        assistantMessage.content += chunk
                    }
                case .info(let info):
                    // Update performance metrics
                    generateCompletionInfo = info
                }
            }
        }

        do {
            // Handle task completion and cancellation
            try await withTaskCancellationHandler {
                try await generateTask?.value
            } onCancel: {
                Task { @MainActor in
                    generateTask?.cancel()

                    // Mark message as cancelled
                    if let assistantMessage = messages.last {
                        assistantMessage.content += "\n[Cancelled]"
                    }
                }
            }
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
            messages = []
            generateTask?.cancel()
        }

        if options.contains(.meta) {
            generateCompletionInfo = nil
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
    var images: [URL] = []

    /// Currently selected video URLs
    var videos: [URL] = []

    /// Whether any media is currently selected
    var isEmpty: Bool {
        images.isEmpty && videos.isEmpty
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

enum MessageDisplayMode: String, CaseIterable, Identifiable {
    case markdown
    case plainText

    var id: String { rawValue }

    /// User-friendly display name for the mode
    var displayName: String {
        switch self {
        case .markdown:
            "Markdown"
        case .plainText:
            "Plain Text"
        }
    }
}
