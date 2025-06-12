// Copyright © 2024 Apple Inc.

import MLX
import MLXLMCommon
import Tokenizers

/// Marker protocol for LLMModels
public protocol LLMModel: LanguageModel, LoRAModel {

    /// Models can implement this is they need a custom `MessageGenerator`.
    ///
    /// The default implementation returns `DefaultMessageGenerator`.
    func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator
}

extension LLMModel {

    /// Default prepare step for ``LLMModel``.
    ///
    /// This will evaluate the prompt in chunks until there is a small amount of
    /// tokens left to feed into the `TokenIterator`.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512
        var y = input.text
        var state: LMOutput.State? = nil

        // prepare the prompt in chunks if larger than the prefill size
        while y.tokens.size > prefillStepSize {
            let input = y[.newAxis, ..<prefillStepSize]
            let result = self(input, cache: cache.isEmpty ? nil : cache, state: state)
            eval(cache)
            y = y[prefillStepSize...]
        }

        return .tokens(y)
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
