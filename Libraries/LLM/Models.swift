// Copyright © 2024 Apple Inc.

import Foundation
import Hub

/// Registry of models and and any overrides that go with them, e.g. prompt augmentation.
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

    /// custom preparation logic for the prompt.  custom tokenizers provide more capability, but this
    /// allows some minor formtting changes, e.g. wrapping the user input in the expected prompt
    /// format
    private let preparePrompt: (@Sendable (String) -> String)?

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
        self.preparePrompt = preparePrompt
    }

    public init(
        directory: URL, tokenizerId: String? = nil, overrideTokenizer: String? = nil,
        defaultPrompt: String = "hello",
        extraEOSTokens: Set<String> = [],
        preparePrompt: (@Sendable (String) -> String)? = nil
    ) {
        self.id = .directory(directory)
        self.tokenizerId = tokenizerId
        self.overrideTokenizer = overrideTokenizer
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.preparePrompt = preparePrompt
    }

    public func prepare(prompt: String) -> String {
        preparePrompt?(prompt) ?? prompt
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
    ) {
        prompt in
        "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
    }

    public static let mistralNeMo4bit = ModelConfiguration(
        id: "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        defaultPrompt: "Explain quaternions."
    ) { prompt in
        "<s>[INST] \(prompt) [/INST] "
    }

    public static let mistral7B4bit = ModelConfiguration(
        id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        defaultPrompt: "Describe the Swift language."
    ) { prompt in
        "<s>[INST] \(prompt) [/INST] "
    }

    public static let codeLlama13b4bit = ModelConfiguration(
        id: "mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "func sortArray(_ array: [Int]) -> String { <FILL_ME> }"
    ) { prompt in
        // given the prompt: func sortArray(_ array: [Int]) -> String { <FILL_ME> }
        // the python code produces this (via its custom tokenizer):
        // <PRE> func sortArray(_ array: [Int]) -> String {  <SUF> } <MID>

        "<PRE> " + prompt.replacingOccurrences(of: "<FILL_ME>", with: "<SUF>") + " <MID>"
    }

    public static let phi4bit = ModelConfiguration(
        id: "mlx-community/phi-2-hf-4bit-mlx",

        // https://www.promptingguide.ai/models/phi-2
        defaultPrompt: "Why is the sky blue?"
    )

    public static let phi3_5_4bit = ModelConfiguration(
        id: "mlx-community/Phi-3.5-mini-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    ) {
        prompt in
        "<s><|user|>\n\(prompt)<|end|>\n<|assistant|>\n"
    }

    public static let gemma2bQuantized = ModelConfiguration(
        id: "mlx-community/quantized-gemma-2b-it",
        overrideTokenizer: "PreTrainedTokenizer",

        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "what is the difference between lettuce and cabbage?"

    ) { prompt in
        "<start_of_turn>user\n\(prompt)<end_of_turn>\n<start_of_turn>model\n"
    }

    public static let gemma_2_9b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-2-9b-it-4bit",
        overrideTokenizer: "PreTrainedTokenizer",

        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "What is the difference between lettuce and cabbage?"

    ) { prompt in
        "<start_of_turn>user\n\(prompt)<end_of_turn>\n<start_of_turn>model\n"
    }

    public static let gemma_2_2b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-2-2b-it-4bit",
        overrideTokenizer: "PreTrainedTokenizer",

        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "What is the difference between lettuce and cabbage?"

    ) { prompt in
        "<start_of_turn>user \(prompt)<end_of_turn><start_of_turn>model"
    }

    public static let qwen205b4bit = ModelConfiguration(
        id: "mlx-community/Qwen1.5-0.5B-Chat-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "why is the sky blue?"
    ) { prompt in
        "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant"
    }

    public static let openelm270m4bit = ModelConfiguration(
        id: "mlx-community/OpenELM-270M-Instruct",

        // https://huggingface.co/apple/OpenELM
        defaultPrompt: "Once upon a time there was"
    ) { prompt in
        "\(prompt)"
    }

    public static let llama3_1_8B_4bit = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    ) {
        prompt in
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\(prompt)<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    }

    public static let llama3_8B_4bit = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    ) {
        prompt in
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\(prompt)<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    }

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
                llama3_1_8B_4bit,
                mistralNeMo4bit,
                smolLM_135M_4bit,
                mistral7B4bit,
                codeLlama13b4bit,
                phi4bit,
                phi3_5_4bit,
                gemma2bQuantized,
                gemma_2_9b_it_4bit,
                qwen205b4bit,
                openelm270m4bit,
            ])
            bootstrapState = .bootstrapped

        case .bootstrapping:
            break

        case .bootstrapped:
            break
        }
    }
}
