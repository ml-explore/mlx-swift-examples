//
//  LMModel.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 21.04.2025.
//

import MLXLMCommon

/// Represents a language model configuration with its associated properties and type.
/// Can represent either a large language model (LLM) or a vision-language model (VLM).
struct LMModel {
    /// Name of the model
    let name: String

    /// Configuration settings for model initialization
    let configuration: ModelConfiguration

    /// Type of the model (language or vision-language)
    let type: ModelType

    /// Defines the type of language model
    enum ModelType {
        /// Large language model (text-only)
        case llm
        /// Vision-language model (supports images and text)
        case vlm
    }
}

// MARK: - Helpers

extension LMModel {
    /// Display name with additional "(Vision)" suffix for vision models
    var displayName: String {
        if isVisionModel {
            "\(name) (Vision)"
        } else {
            name
        }
    }

    /// Whether the model is a large language model
    var isLanguageModel: Bool {
        type == .llm
    }

    /// Whether the model is a vision-language model
    var isVisionModel: Bool {
        type == .vlm
    }
}

extension LMModel: Identifiable, Hashable {
    var id: String {
        name
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(name)
    }
}
