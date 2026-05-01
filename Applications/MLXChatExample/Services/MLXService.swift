//
//  MLXService.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

/// A service class that manages machine learning models for text and vision-language tasks.
/// This class handles model loading, caching, and text generation using various LLM and VLM models.
@Observable
class MLXService {
    /// List of available models that can be used for generation.
    /// Includes both language models (LLM) and vision-language models (VLM).
    static let availableModels: [LMModel] = [
        LMModel(
            name: "llama3.2:1b", displayName: "Llama 3.2 (1B)",
            configuration: LLMRegistry.llama3_2_1B_4bit, type: .llm),
        LMModel(
            name: "qwen2.5:1.5b", displayName: "Qwen 2.5 (1.5B)",
            configuration: LLMRegistry.qwen2_5_1_5b, type: .llm),
        LMModel(
            name: "smolLM:135m", displayName: "SmolLM (135M)",
            configuration: LLMRegistry.smolLM_135M_4bit, type: .llm),
        LMModel(
            name: "qwen3:0.6b", displayName: "Qwen 3 (0.6B)",
            configuration: LLMRegistry.qwen3_0_6b_4bit, type: .llm),
        LMModel(
            name: "qwen3:1.7b", displayName: "Qwen 3 (1.7B)",
            configuration: LLMRegistry.qwen3_1_7b_4bit, type: .llm),
        LMModel(
            name: "qwen3:4b", displayName: "Qwen 3 (4B)",
            configuration: LLMRegistry.qwen3_4b_4bit, type: .llm),
        LMModel(
            name: "qwen3:8b", displayName: "Qwen 3 (8B)",
            configuration: LLMRegistry.qwen3_8b_4bit, type: .llm),
        LMModel(
            name: "qwen2.5VL:3b", displayName: "Qwen 2.5 VL (3B)",
            configuration: VLMRegistry.qwen2_5VL3BInstruct4Bit, type: .vlm),
        LMModel(
            name: "qwen2VL:2b", displayName: "Qwen 2 VL (2B)",
            configuration: VLMRegistry.qwen2VL2BInstruct4Bit, type: .vlm),
        LMModel(
            name: "smolVLM", displayName: "SmolVLM",
            configuration: VLMRegistry.smolvlminstruct4bit, type: .vlm),
        LMModel(
            name: "acereason:7B", displayName: "AceReason (7B)",
            configuration: LLMRegistry.acereason_7b_4bit, type: .llm),
        LMModel(
            name: "gemma3n:E2B", displayName: "Gemma 3n (E2B)",
            configuration: LLMRegistry.gemma3n_E2B_it_lm_4bit, type: .llm),
        LMModel(
            name: "gemma3n:E4B", displayName: "Gemma 3n (E4B)",
            configuration: LLMRegistry.gemma3n_E4B_it_lm_4bit, type: .llm),
    ]

    /// Cache to store loaded model containers to avoid reloading.
    private let modelCache = NSCache<NSString, ModelContainer>()

    /// Tracks the current model download progress.
    /// Non-nil only while bytes are actively flowing from the network.
    @MainActor
    private(set) var modelDownloadProgress: Progress?

    /// Whether a model is currently being initialized into memory.
    /// True for the entire `loadContainer` call, including the disk → memory phase
    /// after any network download has finished.
    @MainActor
    private(set) var isLoadingModel = false

    /// Loads a model from the hub or retrieves it from cache.
    /// - Parameter model: The model configuration to load
    /// - Returns: A ModelContainer instance containing the loaded model
    /// - Throws: Errors that might occur during model loading
    private func load(model: LMModel) async throws -> ModelContainer {
        // Set GPU memory limit to prevent out of memory issues
        Memory.cacheLimit = 20 * 1024 * 1024

        // Return cached model if available to avoid reloading
        if let container = modelCache.object(forKey: model.name as NSString) {
            return container
        } else {
            // Select appropriate factory based on model type
            let factory: any ModelFactory =
                switch model.type {
                case .llm:
                    LLMModelFactory.shared
                case .vlm:
                    VLMModelFactory.shared
                }

            let downloader = #hubDownloader()
            let loader = #huggingFaceTokenizerLoader()

            await MainActor.run { self.isLoadingModel = true }
            defer {
                Task { @MainActor in
                    self.modelDownloadProgress = nil
                    self.isLoadingModel = false
                }
            }

            // Load model and track download progress. Clear the progress as soon as
            // the downloader reports completion so the loading-into-memory phase is
            // represented by `isLoadingModel` alone.
            let container = try await factory.loadContainer(
                from: downloader,
                using: loader,
                configuration: model.configuration
            ) { progress in
                Task { @MainActor in
                    guard self.isLoadingModel else { return }
                    if progress.isFinished {
                        self.modelDownloadProgress = nil
                    } else {
                        self.modelDownloadProgress = progress
                    }
                }
            }

            // Cache the loaded model for future use
            modelCache.setObject(container, forKey: model.name as NSString)

            return container
        }
    }

    /// Generates text based on the provided messages using the specified model.
    /// - Parameters:
    ///   - messages: Array of chat messages including user, assistant, and system messages
    ///   - model: The language model to use for generation
    /// - Returns: An AsyncStream of generated text tokens
    /// - Throws: Errors that might occur during generation
    func generate(messages: [Message], model: LMModel) async throws -> AsyncStream<Generation> {
        // Load or retrieve model from cache
        let modelContainer = try await load(model: model)

        // Map app-specific Message type to Chat.Message for model input
        let chat = messages.map { message in
            let role: Chat.Message.Role =
                switch message.role {
                case .assistant:
                    .assistant
                case .user:
                    .user
                case .system:
                    .system
                }

            // Process any attached media for VLM models
            let images: [UserInput.Image] = message.images.map { imageURL in .url(imageURL) }
            let videos: [UserInput.Video] = message.videos.map { videoURL in .url(videoURL) }

            return Chat.Message(
                role: role, content: message.content, images: images, videos: videos)
        }

        // Prepare input for model processing
        let userInput = UserInput(
            chat: chat, processing: .init(resize: .init(width: 1024, height: 1024)))

        // Generate response using the model
        return try await modelContainer.perform(nonSendable: userInput) {
            (context: ModelContext, userInput: UserInput) in
            let lmInput = try await context.processor.prepare(input: userInput)
            // Set temperature for response randomness (0.7 provides good balance)
            let parameters = GenerateParameters(temperature: 0.7)

            return try MLXLMCommon.generate(
                input: lmInput, parameters: parameters, context: context)
        }
    }
}
