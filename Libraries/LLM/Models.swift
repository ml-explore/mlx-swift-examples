// Copyright © 2024 Apple Inc.

import Foundation
import Hub

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The python tokenizers have a very rich set of implementations and configuration.  The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
public struct ModelConfiguration: Sendable {

    public enum Identifier: Sendable {
        case id(String)
        case directory(URL)
    }

    public var id: Identifier

    public var name: String {
        switch id {
        case .id(let string):
            string
        case .directory(let url):
            url.deletingLastPathComponent().lastPathComponent + "/" + url.lastPathComponent
        }
    }

    /// pull the tokenizer from an alternate id
    public let tokenizerId: String?

    /// overrides for TokenizerModel/knownTokenizers -- useful before swift-transformers is updated
    public let overrideTokenizer: String?

    /// A reasonable default prompt for the model
    public let defaultPrompt: String

    /// Additional tokens to use for end of string
    public let extraEOSTokens: Set<String>

    public init(
        id: String, tokenizerId: String? = nil, overrideTokenizer: String? = nil,
        defaultPrompt: String = "hello",
        extraEOSTokens: Set<String> = [],
        preparePrompt: (@Sendable (String) -> String)? = nil
    ) {
        self.id = .id(id)
        self.tokenizerId = tokenizerId
        self.overrideTokenizer = overrideTokenizer
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
    }

    public init(
        directory: URL, tokenizerId: String? = nil, overrideTokenizer: String? = nil,
        defaultPrompt: String = "hello",
        extraEOSTokens: Set<String> = []
    ) {
        self.id = .directory(directory)
        self.tokenizerId = tokenizerId
        self.overrideTokenizer = overrideTokenizer
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
    }

    public func modelDirectory(hub: HubApi = HubApi()) -> URL {
        switch id {
        case .id(let id):
            // download the model weights and config
            let repo = Hub.Repo(id: id)
            return hub.localRepoLocation(repo)

        case .directory(let directory):
            return directory
        }
    }

    @MainActor
    public static var registry = [String: ModelConfiguration]()

    @MainActor
    public static func register(configurations: [ModelConfiguration]) {
        bootstrap()

        for c in configurations {
            registry[c.name] = c
        }
    }

    @MainActor
    public static func configuration(id: String) -> ModelConfiguration {
        bootstrap()

        if let c = registry[id] {
            return c
        } else {
            return ModelConfiguration(id: id)
        }
    }
}

extension ModelConfiguration {
    public static let smolLM_135M_4bit = ModelConfiguration(
        id: "mlx-community/SmolLM-135M-Instruct-4bit",
        defaultPrompt: "Tell me about the history of Spain."
    )

    public static let mistralNeMo4bit = ModelConfiguration(
        id: "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        defaultPrompt: "Explain quaternions."
    )

    public static let mistral7B4bit = ModelConfiguration(
        id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        defaultPrompt: "Describe the Swift language."
    )

    public static let codeLlama13b4bit = ModelConfiguration(
        id: "mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "func sortArray(_ array: [Int]) -> String { <FILL_ME> }"
    )

    public static let phi4bit = ModelConfiguration(
        id: "mlx-community/phi-2-hf-4bit-mlx",
        // https://www.promptingguide.ai/models/phi-2
        defaultPrompt: "Why is the sky blue?"
    )

    public static let phi3_5_4bit = ModelConfiguration(
        id: "mlx-community/Phi-3.5-mini-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    )

    public static let phi3_5MoE = ModelConfiguration(
        id: "mlx-community/Phi-3.5-MoE-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    ) {
        prompt in
        "<|user|>\n\(prompt)<|end|>\n<|assistant|>\n"
    }

    public static let gemma2bQuantized = ModelConfiguration(
        id: "mlx-community/quantized-gemma-2b-it",
        overrideTokenizer: "PreTrainedTokenizer",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "what is the difference between lettuce and cabbage?"
    )

    public static let gemma_2_9b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-2-9b-it-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    public static let gemma_2_2b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-2-2b-it-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    public static let qwen205b4bit = ModelConfiguration(
        id: "mlx-community/Qwen1.5-0.5B-Chat-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "why is the sky blue?"
    )

    public static let openelm270m4bit = ModelConfiguration(
        id: "mlx-community/OpenELM-270M-Instruct",
        // https://huggingface.co/apple/OpenELM
        defaultPrompt: "Once upon a time there was"
    )

    public static let llama3_1_8B_4bit = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    public static let llama3_8B_4bit = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    public static let llama3_2_1B_4bit = ModelConfiguration(
        id: "mlx-community/Llama-3.2-1B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    public static let llama3_2_3B_4bit = ModelConfiguration(
        id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    private enum BootstrapState: Sendable {
        case idle
        case bootstrapping
        case bootstrapped
    }

    @MainActor
    static private var bootstrapState = BootstrapState.idle

    @MainActor
    static func bootstrap() {
        switch bootstrapState {
        case .idle:
            bootstrapState = .bootstrapping
            register(configurations: [
                codeLlama13b4bit,
                gemma2bQuantized,
                gemma_2_2b_it_4bit,
                gemma_2_9b_it_4bit,
                llama3_1_8B_4bit,
                llama3_2_1B_4bit,
                llama3_2_3B_4bit,
                llama3_8B_4bit,
                mistral7B4bit,
                mistralNeMo4bit,
                openelm270m4bit,
                phi3_5MoE,
                phi3_5_4bit,
                phi4bit,
                qwen205b4bit,
                smolLM_135M_4bit,
            ])
            bootstrapState = .bootstrapped

        case .bootstrapping:
            break

        case .bootstrapped:
            break
        }
    }
}
