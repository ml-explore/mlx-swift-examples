//
//  MLXService.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

/// A service class that manages machine learning models for text and vision-language tasks.
/// This class handles model loading, caching, and text generation using various LLM and VLM models.
@Observable
class MLXService {
    /// List of available models that can be used for generation.
    /// Includes both language models (LLM) and vision-language models (VLM).
    static let availableModels: [LMModel] = [
        LMModel(name: "llama3.2:1b", configuration: LLMRegistry.llama3_2_1B_4bit, type: .llm),
        LMModel(name: "llama3.2:3b", configuration: LLMRegistry.llama3_2_3B_4bit, type: .llm),
        LMModel(name: "qwen2.5:1.5b", configuration: LLMRegistry.qwen2_5_1_5b, type: .llm),
        LMModel(name: "smolLM:135m", configuration: LLMRegistry.smolLM_135M_4bit, type: .llm),
        LMModel(name: "qwen3:0.6b", configuration: LLMRegistry.qwen3_0_6b_4bit, type: .llm),
        LMModel(name: "qwen3:1.7b", configuration: LLMRegistry.qwen3_1_7b_4bit, type: .llm),
        LMModel(name: "qwen3:4b", configuration: LLMRegistry.qwen3_4b_4bit, type: .llm),
        LMModel(name: "qwen3:8b", configuration: LLMRegistry.qwen3_8b_4bit, type: .llm),
        LMModel(
            name: "qwen2.5VL:3b", configuration: VLMRegistry.qwen2_5VL3BInstruct4Bit, type: .vlm),
        LMModel(name: "qwen2VL:2b", configuration: VLMRegistry.qwen2VL2BInstruct4Bit, type: .vlm),
        LMModel(name: "smolVLM", configuration: VLMRegistry.smolvlminstruct4bit, type: .vlm),
    ]

    /// Cache to store loaded model containers to avoid reloading.
    private let modelCache = NSCache<NSString, ModelContainer>()

    /// Stores a prompt cache for each loaded model
    private let promptCache = NSCache<NSString, PromptCache>()

    /// Tracks the current model download progress.
    /// Access this property to monitor model download status.
    @MainActor
    private(set) var modelDownloadProgress: Progress?

    /// Loads a model from the hub or retrieves it from cache.
    /// - Parameter model: The model configuration to load
    /// - Returns: A ModelContainer instance containing the loaded model
    /// - Throws: Errors that might occur during model loading
    private func load(model: LMModel) async throws -> ModelContainer {
        // Set GPU memory limit to prevent out of memory issues
        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

        // Return cached model if available to avoid reloading
        if let container = modelCache.object(forKey: model.name as NSString) {
            return container
        } else {
            print("Model not loaded \(model.name), loading model...")
            // Select appropriate factory based on model type
            let factory: ModelFactory =
                switch model.type {
                case .llm:
                    LLMModelFactory.shared
                case .vlm:
                    VLMModelFactory.shared
                }

            // Load model and track download progress
            let container = try await factory.loadContainer(
                hub: .default, configuration: model.configuration
            ) { progress in
                Task { @MainActor in
                    self.modelDownloadProgress = progress
                }
            }

            // Clear out the promptCache
            promptCache.removeObject(forKey: model.name as NSString)

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
        let userInput = UserInput(chat: chat)

        // Generate response using the model
        return try await modelContainer.perform { (context: ModelContext) in

            let fullPrompt = try await context.processor.prepare(input: userInput)

            let parameters = GenerateParameters(temperature: 0.7)

            // TODO: Prompt cache access isn't isolated
            // Get the prompt cache and adjust new prompt to remove
            // prefix already in cache, trim cache if cache is
            // inconsistent with new prompt.
            let (cache, lmInput) = getPromptCache(
                fullPrompt: fullPrompt, parameters: parameters, context: context,
                modelName: model.name)

            // TODO: The generated tokens should be added to the prompt cache but not possible with AsyncStream
            return try MLXLMCommon.generate(
                input: lmInput, parameters: parameters, context: context, cache: cache.cache)
        }
    }

    func getPromptCache(
        fullPrompt: LMInput, parameters: GenerateParameters, context: ModelContext,
        modelName: String
    ) -> (PromptCache, LMInput) {
        let cache: PromptCache
        if let existingCache = promptCache.object(forKey: modelName as NSString) {
            cache = existingCache
        } else {
            // Create cache if it doesn't exist yet
            cache = PromptCache(cache: context.model.newCache(parameters: parameters))
            self.promptCache.setObject(cache, forKey: modelName as NSString)
        }

        let lmInput: LMInput

        /// Remove prefix from prompt that is already in cache
        if let suffix = cache.getUncachedSuffix(prompt: fullPrompt.text.tokens) {
            lmInput = LMInput(text: LMInput.Text(tokens: suffix))
        } else {
            // If suffix is nil, the cache is inconsistent with the new prompt
            // and the cache doesn't support trimming so create a new one here.
            let newCache = PromptCache(cache: context.model.newCache(parameters: parameters))
            self.promptCache.setObject(newCache, forKey: modelName as NSString)
            lmInput = fullPrompt
        }

        return (cache, lmInput)
    }
}
