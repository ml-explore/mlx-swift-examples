// Copyright © 2024 Apple Inc.

import Foundation

/// Registry of models and and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The python tokenizers have a very rich set of implementations and configuration.  The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
public struct ModelConfiguration {
    public let id: String

    /// pull the tokenizer from an alternate id
    public let tokenizerId: String?

    /// overrides for TokenizerModel/knownTokenizers -- useful before swift-transformers is updated
    public let overrideTokenizer: String?

    /// custom preparation logic for the prompt.  custom tokenizers provide more capability, but this
    /// allows some minor formtting changes, e.g. wrapping the user input in the expected prompt
    /// format
    private let preparePrompt: ((String) -> String)?

    public init(
        id: String, tokenizerId: String? = nil, overrideTokenizer: String? = nil,
        preparePrompt: ((String) -> String)? = nil
    ) {
        self.id = id
        self.tokenizerId = tokenizerId
        self.overrideTokenizer = overrideTokenizer
        self.preparePrompt = preparePrompt
    }

    public func prepare(prompt: String) -> String {
        preparePrompt?(prompt) ?? prompt
    }

    public static var registry = [String: ModelConfiguration]()

    public static func register(configurations: [ModelConfiguration]) {
        bootstrap()

        for c in configurations {
            registry[c.id] = c
        }
    }

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

    public static let mistral7B4bit = ModelConfiguration(
        id: "mlx-community/Mistral-7B-v0.1-hf-4bit-mlx")

    public static let codeLlama13b4bit = ModelConfiguration(
        id: "mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX",
        overrideTokenizer: "PreTrainedTokenizer"
    ) { prompt in
        // given the prompt: func sortArray(_ array: [Int]) -> String { <FILL_ME> }
        // the python code produces this (via its custom tokenizer):
        // <PRE> func sortArray(_ array: [Int]) -> String {  <SUF> } <MID>

        "<PRE> " + prompt.replacingOccurrences(of: "<FILL_ME>", with: "<SUF>") + " <MID>"
    }

    public static let phi4bit = ModelConfiguration(id: "mlx-community/phi-2-hf-4bit-mlx") {
        prompt in
        "Instruct: \(prompt)\nOutput: "
    }

    public static let gemma2bQuantized = ModelConfiguration(
        id: "mlx-community/quantized-gemma-2b-it",
        overrideTokenizer: "PreTrainedTokenizer"
    ) { prompt in
        "<start_of_turn>user \(prompt)<end_of_turn><start_of_turn>model"
    }

    public static let qwen205b4bit = ModelConfiguration(
        id: "mlx-community/Qwen1.5-0.5B-Chat-4bit",
        overrideTokenizer: "PreTrainedTokenizer"
    ) { prompt in
        "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant"
    }

    private enum BootstrapState {
        case idle
        case bootstrapping
        case bootstrapped
    }

    static private var bootstrapState = BootstrapState.idle

    static func bootstrap() {
        switch bootstrapState {
        case .idle:
            bootstrapState = .bootstrapping
            register(configurations: [
                mistral7B4bit,
                codeLlama13b4bit,
                phi4bit,
                gemma2bQuantized,
                qwen205b4bit,
            ])
            bootstrapState = .bootstrapped

        case .bootstrapping:
            break

        case .bootstrapped:
            break
        }
    }
}
