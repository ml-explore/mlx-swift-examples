// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import Tokenizers

/// A `LogitSampler` is responsible for sampling `logits` produced by
/// a ``LanguageModel`` to produce a token.
///
/// See also: ``LogitProcessor``
public protocol LogitSampler: Sendable {

    /// Given `logits` produce a new `MLXArray` with the token.
    func sample(logits: MLXArray) -> MLXArray
}

/// A `LogitProcessor` is an optional visitor of `logits`.
///
/// The ``LogitProcessor`` is called with the input (prompt) before generating tokens:
///
/// ```swift
/// processor?.prompt(input.text.tokens)
/// ```
///
/// Then for each token generated it has a chance to adjust the logits:
///
/// ```swift
/// logits = processor?.process(logits: logits) ?? logits
/// let y = sampler.sample(logits: logits)
/// processor?.didSample(token: y)
/// ```
///
/// See also: ``LogitSampler``
public protocol LogitProcessor: Sendable {

    /// called before token generation starts with the text tokens of the prompt
    mutating func prompt(_ prompt: MLXArray)

    /// called to visit ad possibly modify the logits
    func process(logits: MLXArray) -> MLXArray

    /// called to provide the sampled token
    mutating func didSample(token: MLXArray)
}

/// Parameters for text generation, see ``TokenIterator``.
///
/// This produces:
///
/// - ``LogitSampler``
/// - ``LogitProcessor``
///
/// for the `TokenIterator`.
public struct GenerateParameters: Sendable {

    /// Step size for processing the prompt
    public var prefillStepSize = 512

    /// Maximum tokens to generate
    public var maxTokens: Int?

    /// Maximum size of the key-value cache. Old entries (except the first 4 tokens) will be overwritten.
    /// When set, uses ``RotatingKVCache`` instead of ``KVCacheSimple``
    public var maxKVSize: Int?

    /// Number of bits to use for KV cache quantization. nil implies no cache quantization.
    public var kvBits: Int?

    /// Group size for KV cache quantization (default: 64)
    public var kvGroupSize: Int = 64

    /// Step to begin using a quantized KV cache when kvBits is non-nil (default: 0)
    public var quantizedKVStart: Int = 0

    /// sampling temperature
    public var temperature: Float = 0.6

    /// top p sampling
    public var topP: Float = 1.0

    /// penalty factor for repeating tokens
    public var repetitionPenalty: Float?

    /// number of tokens to consider for repetition penalty
    public var repetitionContextSize: Int = 20

    public init(
        maxTokens: Int? = nil,
        maxKVSize: Int? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        temperature: Float = 0.6, topP: Float = 1.0, repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20
    ) {
        self.maxTokens = maxTokens
        self.maxKVSize = maxKVSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }

    public func sampler() -> LogitSampler {
        if temperature == 0 {
            return ArgMaxSampler()
        } else if topP > 0 && topP < 1 {
            return TopPSampler(temperature: temperature, topP: topP)
        } else {
            return CategoricalSampler(temperature: temperature)
        }
    }

    public func processor() -> LogitProcessor? {
        if let repetitionPenalty, repetitionContextSize > 0 {
            return RepetitionContext(
                repetitionPenalty: repetitionPenalty, repetitionContextSize: repetitionContextSize)
        } else {
            return nil
        }
    }
}

/// Sampler that uses `argMax` (most likely) to sample the logits.
public struct ArgMaxSampler: LogitSampler {
    public func sample(logits: MLX.MLXArray) -> MLX.MLXArray {
        argMax(logits, axis: -1)
    }
}

/// Sampler that uses `topP` and `temperature` to sample the logits.
public struct TopPSampler: LogitSampler {
    let temp: MLXArray
    let topP: MLXArray
    let randomState: MLXRandom.RandomState

    public init(temperature: Float, topP: Float) {
        self.temp = MLXArray(temperature)
        self.topP = MLXArray(topP)
        self.randomState = MLXRandom.RandomState()
    }

    public func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        return withRandomState(randomState) {
            let probs = softmax(logits / temp, axis: -1)
            let sortedIndices = argSort(probs, axis: -1)

            // probs shape is [B,V] and after take it will be [1, B, V], so we squeeze it back to [B, V]
            let sortedProbs = take(probs, sortedIndices, axis: -1).squeezed(axis: 0)

            let cumulativeProbs = cumsum(sortedProbs, axis: -1)

            let topProbs = MLX.where(
                cumulativeProbs .> (1 - topP), sortedProbs, zeros(like: sortedProbs))

            let sortedToken = categorical(log(topProbs))
            return sortedIndices.squeezed(axis: 0)[sortedToken]
        }
    }
}

/// Processor that uses `temperature` to sample the logits
public struct CategoricalSampler: LogitSampler {
    let temp: MLXArray
    let randomState: MLXRandom.RandomState

    public init(temperature: Float) {
        self.temp = MLXArray(temperature)
        self.randomState = MLXRandom.RandomState()
    }

    public func sample(logits: MLXArray) -> MLXArray {
        return withRandomState(randomState) {
            categorical(logits * (1 / temp))
        }
    }
}

/// Processor that implements a `repetitionPenalty`
public struct RepetitionContext: LogitProcessor {
    /// tokens in the repetition context sliding window
    var tokens = [Int]()

    /// current write into into the tokens circular array
    var index = 0

    /// penalty factor for repeating tokens
    let repetitionPenalty: Float

    /// number of tokens to consider for repetition penalty
    let repetitionContextSize: Int

    public init(repetitionPenalty: Float, repetitionContextSize: Int) {
        precondition(repetitionContextSize > 0)
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }

    mutating public func prompt(_ prompt: MLXArray) {
        if prompt.shape[0] <= repetitionContextSize {
            self.tokens = prompt.asArray(Int.self)
        } else {
            self.tokens = prompt[(-repetitionContextSize)...].asArray(Int.self)
        }
    }

    public func process(logits: MLXArray) -> MLXArray {
        if tokens.count > 0 {
            let indices = MLXArray(tokens.map { UInt32($0) })
            var selectedLogits = logits[0..., indices]

            selectedLogits = MLX.where(
                selectedLogits .< 0, selectedLogits * repetitionPenalty,
                selectedLogits / repetitionPenalty)

            logits[0..., indices] = selectedLogits
            return logits
        }

        return logits
    }

    mutating public func didSample(token: MLXArray) {
        if tokens.count >= repetitionContextSize {
            tokens[index] = token.item(Int.self)
            index = (index + 1) % repetitionContextSize
        } else {
            tokens.append(token.item(Int.self))
        }
    }
}

/// Generator of tokens.
///
/// This is typically used via a call to ``generate(input:parameters:context:didGenerate:)``.
///
/// To use it directly:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: LMInput
/// let model: LanguageModel
///
/// let iterator = try TokenIterator(input: input, model: model, parameters: parameters)
///
/// for token in iterator {
///     ...
/// }
/// ```
///
/// Tokens are integers that can be passed through a `Tokenizer` or ``StreamingDetokenizer`` to produce Strings.
///
/// Port of `generate_step()` from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py
///
/// Note: this uses `asyncEval()` and there may be an async evaluation running after a call to `next()`.
public struct TokenIterator: Sequence, IteratorProtocol {
    let model: any LanguageModel
    var state: LMOutput.State?

    var y: LMInput.Text
    var cache: [KVCache]
    var processor: LogitProcessor?
    let sampler: LogitSampler

    var tokenCount = 0
    let maxTokens: Int?

    // Cache quantization parameters
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int

    /// Initialize a `TokenIterator` with the given tokens. Note: this has been
    /// replaced with ``init(input:model:cache:parameters:)``.
    ///
    /// - Parameters:
    ///   - prompt: the prompt tokens
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - parameters: the generation parameters
    @available(*, deprecated, message: "please use init(input:model:cache:parameters:)")
    public init(
        prompt: MLXArray, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        self.model = model
        self.y = .init(tokens: prompt)
        self.cache = cache ?? model.newCache(parameters: parameters)

        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens

        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart

        try prepare(input: .init(text: y), windowSize: parameters.prefillStepSize)
    }

    /// Initialize a `TokenIterator` with the given input.
    ///
    /// If more control is needed over the generation,
    /// ``init(input:model:cache:processor:sampler:prefillStepSize:)``
    /// allows a caller to specify ``LogitProcessor`` and ``LogitSampler``
    /// directly.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - parameters: the generation parameters
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        self.model = model
        self.y = input.text
        self.cache = cache ?? model.newCache(parameters: parameters)

        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens

        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart

        try prepare(input: input, windowSize: parameters.prefillStepSize)
    }

    /// Initialize a `TokenIterator` with the given input and logit handling.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - processor: the logit processor
    ///   - sampler: the logit sampler
    ///   - prefillStepSize: optional prefill step size
    ///   - maxTokens: maximum number of tokens to generate
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        processor: LogitProcessor?, sampler: LogitSampler, prefillStepSize: Int = 512,
        maxTokens: Int? = nil
    ) throws {
        self.model = model
        self.y = input.text
        self.cache = cache ?? model.newCache(parameters: nil)

        self.processor = processor
        self.sampler = sampler
        self.maxTokens = maxTokens

        // No cache quantization for this direct initialization
        self.kvBits = nil
        self.kvGroupSize = 64
        self.quantizedKVStart = 0

        try prepare(input: input, windowSize: prefillStepSize)
    }

    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        switch try model.prepare(input, cache: cache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens

            // evaluate the remainder of the prompt -- this primes the pump
            let token = step(previous: y)
            y = .init(tokens: token)
            asyncEval(y.tokens)

        case .logits(let result):
            y = .init(tokens: convertToToken(logits: result.logits))
            asyncEval(y.tokens)

            break
        }
    }

    mutating func convertToToken(logits: MLXArray) -> MLXArray {
        // process the logits (one hot array of possible tokens)
        var logits = logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits

        // transform logits back to a token
        let y = sampler.sample(logits: logits)

        processor?.didSample(token: y)

        return y
    }

    /// Evaluate the next token and return the new token (y), updating cache state
    mutating func step(previous: LMInput.Text) -> MLXArray {
        let result = model(
            previous[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
        self.state = result.state

        // Apply dynamic cache quantization after each step
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart
        )

        return convertToToken(logits: result.logits)
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // save current value -- this will be returned
        let previousY = y

        // compute the next state and async eval the next token
        let token = step(previous: previousY)
        y = .init(tokens: token)
        asyncEval(token)

        tokenCount += 1

        return previousY.tokens.item(Int.self)
    }
}

/// Result of a call to ``generate(input:parameters:context:didGenerate:)``.
public struct GenerateResult: Sendable {

    /// Initializes a new `GenerateResult` instance.
    ///
    /// - Parameters:
    ///   - inputText: The input text used for generation.
    ///   - tokens: The array of tokens generated.
    ///   - output: The generated output string.
    ///   - promptTime: The time taken to prompt the input.
    ///   - generateTime: The time taken to generate the output.
    public init(
        inputText: LMInput.Text, tokens: [Int], output: String, promptTime: TimeInterval,
        generateTime: TimeInterval
    ) {
        self.inputText = inputText
        self.tokens = tokens
        self.output = output
        self.promptTime = promptTime
        self.generateTime = generateTime
    }

    /// input (prompt, images, etc.)
    public let inputText: LMInput.Text

    @available(*, deprecated, message: "use inputText")
    public var promptTokens: [Int] {
        inputText.tokens.asArray(Int.self)
    }

    /// output tokens
    public let tokens: [Int]

    /// output text
    public let output: String

    /// The number of tokens included in the input prompt.
    public var promptTokenCount: Int { inputText.tokens.size }

    /// The number of tokens generated by the language model.
    public var generationTokenCount: Int { tokens.count }

    /// time to process the prompt / generate the first token
    public let promptTime: TimeInterval

    /// time to generate the remaining tokens
    public let generateTime: TimeInterval

    /// The number of tokens processed per second during the prompt phase.
    public var promptTokensPerSecond: Double {
        Double(inputText.tokens.size) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    public var tokensPerSecond: Double {
        Double(tokens.count) / generateTime
    }

    public func summary() -> String {
        """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
    }
}

/// Action from token visitor callback in ``generate(input:parameters:context:didGenerate:)``.
public enum GenerateDisposition: Sendable {
    /// keep producing tokens until an EOS token is produced
    case more

    /// stop producing tokens, e.g. a token limit has been hit
    case stop
}

/// Given prompt tokens generate text using the given model and parameters.
///
/// ``generate(input:parameters:context:didGenerate:)`` is the preferred call.
///
/// - Parameters:
///   - promptTokens: tokenized prompt
///   - parameters: generation parameters
///   - model: model to evaluate
///   - tokenizer: tokenizer to convert tokens back into strings and recognizer special tokens
///   - extraEOSTokens: any additional stop tokens
///   - didGenerate: visitor for the tokens as they are generated
@available(*, deprecated, message: "please use generate(input:parameters:context:didGenerate:)")
public func generate(
    promptTokens: [Int], parameters: GenerateParameters, model: any LanguageModel,
    tokenizer: Tokenizer,
    extraEOSTokens: Set<String>? = nil,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let tokens = MLXArray(promptTokens)
    let iterator = try TokenIterator(
        prompt: tokens, model: model, parameters: parameters)

    // this is a compatibility cover -- create the required values
    // for the iteration
    let input = LMInput(tokens: tokens)
    let configuration = ModelConfiguration(id: "stand-in", extraEOSTokens: extraEOSTokens ?? [])
    let context = ModelContext(
        configuration: configuration, model: model, processor: StandInUserInputProcessor(),
        tokenizer: tokenizer)

    return generate(
        input: input, context: context, iterator: iterator, didGenerate: didGenerate)
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// For example:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: UserInput
/// let context: ModelContext
///
/// let lmInput = try context.processor.prepare(input: input)
/// let result = generate(input: lmInput,
///     parameters: generateParameters,
///     context: context) { tokens in
///     .more
/// }
/// ```
///
/// Internally this constructs a ``TokenIterator`` and calls
/// ``generate(input:context:iterator:didGenerate:)``
///
/// - Parameters:
///   - input: prepared language model input
///   - parameters: parameters controlling the token generation
///   - context: model context (model and tokenizer)
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: the generated output
public func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let iterator = try TokenIterator(
        input: input, model: context.model, parameters: parameters)
    return generate(
        input: input, context: context, iterator: iterator, didGenerate: didGenerate)
}

/// Low level token generation using a ``TokenIterator``.
///
/// ``generate(input:parameters:context:didGenerate:)`` is the preferred call.
///
/// - Parameters:
///   - input: prepared language model input
///   - context: model context (model and tokenizer)
///   - iterator: token iterator
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: the generated output
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: ([Int]) -> GenerateDisposition
) -> GenerateResult {
    var start = Date.timeIntervalSinceReferenceDate
    var promptTime: TimeInterval = 0

    let additionalEOSTokenIds = Set(
        (context.configuration.extraEOSTokens ?? [])
            .compactMap {
                context.tokenizer.convertTokenToId($0)
            })

    var tokens = [Int]()

    for token in iterator {
        // compute the timing for the prompt
        if tokens.isEmpty {
            let now = Date.timeIntervalSinceReferenceDate
            promptTime = now - start
            start = now
        }

        if token == context.tokenizer.unknownTokenId || token == context.tokenizer.eosTokenId
            || additionalEOSTokenIds.contains(token)
        {
            break
        }
        tokens.append(token)

        if didGenerate(tokens) == .stop {
            break
        }
    }

    let now = Date.timeIntervalSinceReferenceDate
    let generateTime = now - start

    // TokenIterator uses `asyncEval()` to keep the pipeline full. If the caller
    // exits the program right away, those tasks will still be executing and will
    // hit assertions as the mlx scheduler is torn down. Synchronize with the stream
    // to make sure it is complete.
    Stream().synchronize()

    return GenerateResult(
        inputText: input.text, tokens: tokens,
        output: context.tokenizer.decode(tokens: tokens),
        promptTime: promptTime, generateTime: generateTime)
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// For example:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: UserInput
/// let context: ModelContext
///
/// let lmInput = try context.processor.prepare(input: input)
/// let result = generate(input: lmInput,
///     parameters: generateParameters,
///     context: context) { token in
///     .more
/// }
/// ```
///
/// Internally this constructs a ``TokenIterator`` and calls
/// ``generate(input:context:iterator:didGenerate:)``
///
/// - Parameters:
///   - input: prepared language model input
///   - parameters: parameters controlling the token generation
///   - context: model context (model and tokenizer)
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: Information about the generation
public func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: (Int) -> GenerateDisposition
) throws -> GenerateCompletionInfo {
    let iterator = try TokenIterator(
        input: input, model: context.model, parameters: parameters)
    return generate(
        input: input, context: context, iterator: iterator, didGenerate: didGenerate)
}

public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: (Int) -> GenerateDisposition
) -> GenerateCompletionInfo {
    var start = Date.timeIntervalSinceReferenceDate
    var promptTime: TimeInterval = 0

    let additionalEOSTokenIds = Set(
        (context.configuration.extraEOSTokens ?? [])
            .compactMap {
                context.tokenizer.convertTokenToId($0)
            })

    var tokenCount = 0

    for token in iterator {
        // Compute the timing for the prompt
        if promptTime == 0 {
            let now = Date.timeIntervalSinceReferenceDate
            promptTime = now - start
            start = now
        }

        // Check for end-of-sequence tokens
        if token == context.tokenizer.unknownTokenId || token == context.tokenizer.eosTokenId
            || additionalEOSTokenIds.contains(token)
        {
            break
        }

        tokenCount += 1

        // Invoke the callback with the current token
        if didGenerate(token) == .stop {
            break
        }
    }

    let now = Date.timeIntervalSinceReferenceDate
    let generateTime = now - start

    // Synchronize with the stream to ensure tasks are completed
    Stream().synchronize()

    return GenerateCompletionInfo(
        promptTokenCount: input.text.tokens.size,
        generationTokenCount: tokenCount,
        promptTime: promptTime,
        generationTime: generateTime
    )
}

/// Generates tokens asynchronously using the provided language model input, parameters, and context.
///
/// This function initializes a `TokenIterator` with the given input, model, and generation parameters,
/// and then streams the token generation process via an `AsyncStream`. The resulting stream yields
/// instances of the `Generation` enum, which can represent either individual tokens or summary
/// completion information.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
/// - Returns: An `AsyncStream` that emits `Generation` values, including generated tokens (`.token`)
///   and completion information (`.info`).
/// - Throws: An error if the `TokenIterator` initialization fails due to invalid input or model configuration.
///
/// ### Example Usage:
/// ```swift
/// // Define the input, parameters, and context for token generation.
/// let generateParameters: GenerateParameters
/// let input: UserInput
/// let context: ModelContext
///
/// let lmInput = try context.processor.prepare(input: input)
///
/// // Call the generate function to get an AsyncStream.
/// let stream = try generate(input: lmInput, parameters: parameters, context: context)
///
/// // Process the stream asynchronously to handle generated tokens and completion info.
/// for await generation in stream {
///     switch generation {
///     case .token(let token):
///         print("Generated token: \(context.tokenizer.decode(tokens: [token])")
///     case .info(let info):
///         print("Finished: \(info.tokensPerSecond) tokens/s.")
///     }
/// }
/// ```
public func generate(
    input: LMInput, cache: [KVCache]? = nil, parameters: GenerateParameters, context: ModelContext
) throws -> AsyncStream<Generation> {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters)
    return generate(
        input: input, context: context, iterator: iterator)
}

public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator
) -> AsyncStream<Generation> {

    AsyncStream { continuation in

        // Launch a Task to perform iteration asynchronously.
        let task = Task {
            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0

            let additionalEOSTokenIds = Set(
                context.configuration.extraEOSTokens
                    .compactMap {
                        context.tokenizer.convertTokenToId($0)
                    })

            var tokenCount = 0
            var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
            let toolCallProcessor = ToolCallProcessor()

            for token in iterator {

                // Check for cancellation on every loop iteration.
                if Task.isCancelled { break }

                if promptTime == 0 {
                    let now = Date.timeIntervalSinceReferenceDate
                    promptTime = now - start
                    start = now
                }

                if token == context.tokenizer.unknownTokenId
                    || token == context.tokenizer.eosTokenId
                    || additionalEOSTokenIds.contains(token)
                {
                    break
                }

                detokenizer.append(token: token)
                if let chunk = detokenizer.next() {
                    tokenCount += 1

                    // Process chunk through the tool call processor
                    if let textToYield = toolCallProcessor.processChunk(chunk) {
                        continuation.yield(.chunk(textToYield))
                    }

                    // Check if we have a complete tool call
                    if let toolCall = toolCallProcessor.toolCalls.popLast() {
                        continuation.yield(.toolCall(toolCall))
                    }
                }
            }

            let now = Date.timeIntervalSinceReferenceDate
            let generateTime = now - start

            let info = GenerateCompletionInfo(
                promptTokenCount: input.text.tokens.size,
                generationTokenCount: tokenCount,
                promptTime: promptTime,
                generationTime: generateTime
            )
            continuation.yield(.info(info))

            // Synchronize with the stream to ensure tasks are completed
            Stream().synchronize()

            // Finalize the stream
            continuation.finish()
        }
        // When the consumer cancels (or ends) the stream,
        // cancel our underlying task.
        continuation.onTermination = { _ in
            task.cancel()
        }
    }
}

/// Represents metadata and statistics related to token generation.
///
/// Provides information about the number of tokens processed during both the prompt and generation phases, as well as the time taken for each phase.
public struct GenerateCompletionInfo: Sendable {
    /// The number of tokens included in the input prompt.
    public let promptTokenCount: Int

    /// The number of tokens generated by the language model.
    public let generationTokenCount: Int

    /// The time interval (in seconds) taken to process the input prompt.
    public let promptTime: TimeInterval

    /// The time interval (in seconds) taken to generate the output tokens.
    public let generateTime: TimeInterval

    /// The number of tokens processed per second during the prompt phase.
    public var promptTokensPerSecond: Double {
        Double(promptTokenCount) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    public var tokensPerSecond: Double {
        Double(generationTokenCount) / generateTime
    }

    public init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        promptTime: TimeInterval,
        generationTime: TimeInterval
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTime = promptTime
        self.generateTime = generationTime
    }

    public func summary() -> String {
        """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
    }
}

/// Represents the different stages or outputs of the token generation process.
///
/// This enum distinguishes between the following:
/// - `.chunk`: A decoded string from one or more tokens generated by the language model.
/// - `.info`: Metadata and performance statistics about the generation process.
public enum Generation: Sendable {
    /// A generated token represented as a String
    case chunk(String)

    /// Completion information summarizing token counts and performance metrics.
    case info(GenerateCompletionInfo)

    /// A tool call from the language model.
    case toolCall(ToolCall)

    /// Generated text or nil
    public var chunk: String? {
        switch self {
        case .chunk(let string): string
        case .info: nil
        case .toolCall: nil
        }
    }

    /// Completion info or nil
    public var info: GenerateCompletionInfo? {
        switch self {
        case .chunk: nil
        case .info(let info): info
        case .toolCall: nil
        }
    }

    /// Tool call or nil
    public var toolCall: ToolCall? {
        switch self {
        case .chunk: nil
        case .info: nil
        case .toolCall(let toolCall): toolCall
        }
    }

    /// Reducer that can be used with `throttle()` to gather elements into a batch
    @Sendable
    public static func collect(_ batch: [Generation]?, _ element: Generation) -> [Generation] {
        (batch ?? []) + [element]
    }
}
