//
//  MLXService.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import Foundation
import HuggingFace
import LMResponseParserMLX
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

/// A service class that manages machine learning models for text and vision-language tasks.
/// Holds a single active `ResponseChatSession` so KV-cache state is reused across turns
/// without re-feeding the visible message array. When the selected model changes,
/// the session is rebuilt and seeded with the current visible chat history so the
/// new model can continue the conversation.
@Observable
@MainActor
class MLXService {
    /// List of available models that can be used for generation.
    /// Includes both language models (LLM) and vision-language models (VLM).
    /// `qwen3:4b` is listed first so it is the default selection on a fresh launch.
    static let availableModels: [LMModel] = [
        LMModel(
            name: "qwen3:4b", displayName: "Qwen 3 (4B)",
            configuration: LLMRegistry.qwen3_4b_4bit, type: .llm),
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
        LMModel(
            name: "gpt-oss:20b", displayName: "GPT-OSS (20B, MXFP4 Q8)",
            configuration: LLMRegistry.gpt_oss_20b_MXFP4_Q8, type: .llm),
        LMModel(
            name: "granite4:tiny", displayName: "Granite 4.0 H Tiny (4-bit DWQ)",
            configuration: LLMRegistry.granite_4_0_h_tiny_4bit_dwq, type: .llm),
        LMModel(
            name: "gemma4:E2B", displayName: "Gemma 4 (E2B)",
            configuration: LLMRegistry.gemma4_e2b_it_4bit, type: .llm),
        LMModel(
            name: "lfm2:1.2b", displayName: "LFM2 (1.2B)",
            configuration: LLMRegistry.lfm2_1_2b_4bit, type: .llm),
    ]

    /// System prompt applied to each new `ResponseChatSession` (as `instructions`)
    /// and shown in the chat as the system message.
    static let systemPrompt = "You are a helpful assistant."

    /// Generation parameters applied to each new `ResponseChatSession`.
    private let generateParameters = GenerateParameters(temperature: 0.7)

    /// Cache of loaded model containers, keyed by `LMModel.name`.
    private let modelCache = NSCache<NSString, ModelContainer>()

    /// The currently active session, owning the KV cache for `currentModel`.
    /// Rebuilt with seeded history whenever the selected model changes.
    private var currentSession: ResponseChatSession?

    /// The model that `currentSession` was built for. Used to detect a model
    /// switch on the next call to ``generate(history:prompt:images:videos:model:)``.
    private var currentModel: LMModel?

    /// Output token count from the most recent finalized turn, used by
    /// `ChatViewModel` to derive the toolbar's tokens-per-second readout.
    /// `nil` until the first turn finalizes; `nil` again after a turn
    /// that was cancelled or threw, since `ResponseChatSession.lastResponse`
    /// only updates on clean completion.
    var lastResponseOutputTokens: Int? {
        currentSession?.lastResponse?.usage?.outputTokens
    }

    /// Tracks the current model download progress.
    /// Non-nil only while bytes are actively flowing from the network.
    private(set) var modelDownloadProgress: Progress?

    /// Whether a model is currently being initialized into memory.
    /// True for the entire `loadContainer` call, including the disk → memory phase
    /// after any network download has finished.
    private(set) var isLoadingModel = false

    /// Loads a model from the hub or retrieves it from cache.
    private func load(model: LMModel) async throws -> ModelContainer {
        // Set GPU memory limit to prevent out of memory issues
        Memory.cacheLimit = 20 * 1024 * 1024

        if let container = modelCache.object(forKey: model.name as NSString) {
            return container
        }

        let factory: any ModelFactory =
            switch model.type {
            case .llm:
                LLMModelFactory.shared
            case .vlm:
                VLMModelFactory.shared
            }

        let downloader = #hubDownloader()
        let loader = #huggingFaceTokenizerLoader()

        isLoadingModel = true
        defer {
            modelDownloadProgress = nil
            isLoadingModel = false
        }

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

        modelCache.setObject(container, forKey: model.name as NSString)
        return container
    }

    /// Returns the active `ResponseChatSession`, rebuilding it (seeded with `history`)
    /// whenever the selected model changes or no session is yet active.
    ///
    /// `history` is only consumed on a rebuild; for follow-up turns on the same
    /// model the existing session is reused so KV-cache state is preserved.
    private func session(
        for model: LMModel,
        history: [Chat.Message]
    ) async throws -> ResponseChatSession {
        if let currentSession, currentModel?.id == model.id {
            return currentSession
        }
        let container = try await load(model: model)
        let (modelType, modelConfig) = try await readModelTypeAndConfig(from: container)
        // Infer the format up here using the LMModel's pre-load HF repo
        // name. `LLMModelFactory._load` rewrites the loaded `ModelContext`'s
        // configuration to a `.directory(snapshotURL)` id, so the name
        // `ResponseChatSession` would otherwise read becomes
        // `snapshots/<hash>` and stops matching any name-prefix entry
        // in the parser library's table.
        let format = ResponseFormat.infer(
            modelName: model.configuration.name,
            modelType: modelType ?? "",
            modelConfig: modelConfig ?? [:]
        )
        let session = ResponseChatSession(
            container,
            modelType: modelType,
            modelConfig: modelConfig,
            instructions: Self.systemPrompt,
            history: history,
            generateParameters: generateParameters,
            processing: .init(resize: .init(width: 1024, height: 1024)),
            tools: ToolRegistry.allSpecs,
            toolDispatch: { call in
                .string(await ToolRegistry.dispatch(call))
            },
            format: format
        )
        currentSession = session
        currentModel = model
        return session
    }

    /// Reads `config.json` from the loaded container's snapshot directory
    /// for the `model_type` and raw config dict that `ResponseFormat.infer`
    /// needs.
    ///
    /// - TODO: Drop this helper (and the matching `modelType` /
    ///   `modelConfig` arguments to `ResponseChatSession`) once
    ///   `MLXLMCommon.ModelConfiguration` or `ModelContainer` exposes
    ///   the decoded `model_type` and raw config dict directly – the
    ///   loader already decodes both during `loadContainer` and
    ///   discards them.
    private func readModelTypeAndConfig(
        from container: ModelContainer
    ) async throws -> (modelType: String?, modelConfig: [String: any Sendable]?) {
        let directory = try await container.modelDirectory
        let configURL = directory.appending(path: "config.json")
        let data = try Data(contentsOf: configURL)
        let json = try JSONSerialization.jsonObject(with: data)
        guard let dict = json as? [String: any Sendable] else { return (nil, nil) }
        return (dict["model_type"] as? String, dict)
    }

    /// Streams the model's response to the next prompt/media turn as a
    /// sequence of Responses-API events.
    ///
    /// `history` carries the prior conversation (excluding the system prompt and
    /// the new turn being submitted). It is used to seed a fresh session when the
    /// selected model has changed since the previous call; otherwise the existing
    /// session already knows the prior turns and `history` is ignored.
    func generate(
        history: [Chat.Message],
        prompt: String,
        images: [URL] = [],
        videos: [URL] = [],
        model: LMModel
    ) async throws -> AsyncThrowingStream<ResponseStreamingEvent, Error> {
        let session = try await session(for: model, history: history)
        let userImages = images.map { UserInput.Image.url($0) }
        let userVideos = videos.map { UserInput.Video.url($0) }

        return session.streamResponseEvents(
            prompt: prompt,
            images: userImages,
            videos: userVideos
        )
    }

    /// Drops the active session so the next call to ``generate(history:prompt:images:videos:model:)``
    /// builds a fresh one (with whatever history the caller passes at that time).
    func clearSession() {
        currentSession = nil
        currentModel = nil
    }
}
