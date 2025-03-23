// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN
import MLXRandom

// port of https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/__init__.py

/// Iterator that produces latent images.
///
/// Created by:
///
/// - ``TextToImageGenerator/generateLatents(parameters:)``
/// - ``ImageToImageGenerator/generateLatents(image:parameters:strength:)``
public struct DenoiseIterator: Sequence, IteratorProtocol {

    let sd: StableDiffusion

    var xt: MLXArray

    let conditioning: MLXArray
    let cfgWeight: Float
    let textTime: (MLXArray, MLXArray)?

    var i: Int
    let steps: [(MLXArray, MLXArray)]

    init(
        sd: StableDiffusion, xt: MLXArray, t: Int, conditioning: MLXArray, steps: Int,
        cfgWeight: Float, textTime: (MLXArray, MLXArray)? = nil
    ) {
        self.sd = sd
        self.steps = sd.sampler.timeSteps(steps: steps, start: t, dType: sd.dType)
        self.i = 0
        self.xt = xt
        self.conditioning = conditioning
        self.cfgWeight = cfgWeight
        self.textTime = textTime
    }

    public var underestimatedCount: Int {
        steps.count
    }

    mutating public func next() -> MLXArray? {
        guard i < steps.count else {
            return nil
        }

        let (t, tPrev) = steps[i]
        i += 1

        xt = sd.step(
            xt: xt, t: t, tPrev: tPrev, conditioning: conditioning, cfgWeight: cfgWeight,
            textTime: textTime)
        return xt
    }
}

/// Type for the _decoder_ step.
public typealias ImageDecoder = (MLXArray) -> MLXArray

public protocol ImageGenerator {
    func ensureLoaded()

    /// Return a detached decoder -- this is useful if trying to conserve memory.
    ///
    /// The decoder can be used independently of the ImageGenerator to transform
    /// latents into raster images.
    func detachedDecoder() -> ImageDecoder

    /// the equivalent to the ``detachedDecoder()`` but without the detatching
    func decode(xt: MLXArray) -> MLXArray
}

/// Public interface for transforming a text prompt into an image.
///
/// Steps:
///
/// - ``generateLatents(parameters:)``
/// - evaluate each of the latents from the iterator
/// - ``ImageGenerator/decode(xt:)`` or ``ImageGenerator/detachedDecoder()`` to convert the final latent into an image
/// - use ``Image`` to save the image
public protocol TextToImageGenerator: ImageGenerator {
    func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator
}

/// Public interface for transforming a text prompt into an image.
///
/// Steps:
///
/// - ``generateLatents(image:parameters:strength:)``
/// - evaluate each of the latents from the iterator
/// - ``ImageGenerator/decode(xt:)`` or ``ImageGenerator/detachedDecoder()`` to convert the final latent into an image
/// - use ``Image`` to save the image
public protocol ImageToImageGenerator: ImageGenerator {
    func generateLatents(image: MLXArray, parameters: EvaluateParameters, strength: Float)
        -> DenoiseIterator
}

enum ModelContainerError: LocalizedError {
    /// Unable to create the particular type of model, e.g. it doesn't support image to image
    case unableToCreate(String, String)
    /// When operating in conserveMemory mode, it tried to use a model that had been discarded
    case modelDiscarded

    var errorDescription: String? {
        switch self {
        case .unableToCreate(let modelId, let generatorType):
            return String(
                localized:
                    "Unable to create a \(generatorType) with model ID '\(modelId)'. The model may not support this operation type."
            )
        case .modelDiscarded:
            return String(
                localized:
                    "The model has been discarded to conserve memory and is no longer available. Please recreate the model container."
            )
        }
    }
}

/// Container for models that guarantees single threaded access.
public actor ModelContainer<M> {

    enum State {
        case discarded
        case loaded(M)
    }

    var state: State

    /// if true this will discard the model in ``performTwoStage(first:second:)``
    var conserveMemory = false

    private init(model: M) {
        self.state = .loaded(model)
    }

    /// create a ``ModelContainer`` that supports ``TextToImageGenerator``
    static public func createTextToImageGenerator(
        configuration: StableDiffusionConfiguration, loadConfiguration: LoadConfiguration = .init()
    ) throws -> ModelContainer<TextToImageGenerator> {
        if let model = try configuration.textToImageGenerator(configuration: loadConfiguration) {
            return .init(model: model)
        } else {
            throw ModelContainerError.unableToCreate(configuration.id, "TextToImageGenerator")
        }
    }

    /// create a ``ModelContainer`` that supports ``ImageToImageGenerator``
    static public func createImageToImageGenerator(
        configuration: StableDiffusionConfiguration, loadConfiguration: LoadConfiguration = .init()
    ) throws -> ModelContainer<ImageToImageGenerator> {
        if let model = try configuration.imageToImageGenerator(configuration: loadConfiguration) {
            return .init(model: model)
        } else {
            throw ModelContainerError.unableToCreate(configuration.id, "ImageToImageGenerator")
        }
    }

    public func setConserveMemory(_ conserveMemory: Bool) {
        self.conserveMemory = conserveMemory
    }

    /// Perform an action on the model and/or tokenizer.  Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<R>(_ action: @Sendable (M) throws -> R) throws -> R {
        switch state {
        case .discarded:
            throw ModelContainerError.modelDiscarded
        case .loaded(let m):
            try action(m)
        }
    }

    /// Perform a two stage action where the first stage returns values passed to the second stage.
    ///
    /// If ``setConservativeMemory(_:)`` is `true` this will discard the model in between
    /// the `first` and `second` blocks.  The container will have to be recreated if a caller
    /// wants to use it again.
    ///
    /// If `false` this will just run them in sequence and the container can be reused.
    ///
    /// Callers _must_ eval any `MLXArray` before returning as `MLXArray` is not `Sendable`.
    public func performTwoStage<R1, R2>(
        first: @Sendable (M) throws -> R1, second: @Sendable (R1) throws -> R2
    ) throws -> R2 {
        let r1 =
            switch state {
            case .discarded:
                throw ModelContainerError.modelDiscarded
            case .loaded(let m):
                try first(m)
            }
        if conserveMemory {
            self.state = .discarded
        }
        return try second(r1)
    }

}

/// Base class for Stable Diffusion.
open class StableDiffusion {

    let dType: DType
    let diffusionConfiguration: DiffusionConfiguration
    let unet: UNetModel
    let textEncoder: CLIPTextModel
    let autoencoder: Autoencoder
    let sampler: SimpleEulerSampler
    let tokenizer: CLIPTokenizer

    internal init(
        hub: HubApi, configuration: StableDiffusionConfiguration, dType: DType,
        diffusionConfiguration: DiffusionConfiguration? = nil, unet: UNetModel? = nil,
        textEncoder: CLIPTextModel? = nil, autoencoder: Autoencoder? = nil,
        sampler: SimpleEulerSampler? = nil, tokenizer: CLIPTokenizer? = nil
    ) throws {
        self.dType = dType
        self.diffusionConfiguration =
            try diffusionConfiguration
            ?? loadDiffusionConfiguration(hub: hub, configuration: configuration)
        self.unet = try unet ?? loadUnet(hub: hub, configuration: configuration, dType: dType)
        self.textEncoder =
            try textEncoder ?? loadTextEncoder(hub: hub, configuration: configuration, dType: dType)

        // note: autoencoder uses float32 weights
        self.autoencoder =
            try autoencoder
            ?? loadAutoEncoder(hub: hub, configuration: configuration, dType: .float32)

        if let sampler {
            self.sampler = sampler
        } else {
            self.sampler = SimpleEulerSampler(configuration: self.diffusionConfiguration)
        }
        self.tokenizer = try tokenizer ?? loadTokenizer(hub: hub, configuration: configuration)
    }

    open func ensureLoaded() {
        eval(unet, textEncoder, autoencoder)
    }

    func tokenize(tokenizer: CLIPTokenizer, text: String, negativeText: String?) -> MLXArray {
        var tokens = [tokenizer.tokenize(text: text)]
        if let negativeText {
            tokens.append(tokenizer.tokenize(text: negativeText))
        }

        let c = tokens.count
        let max = tokens.map { $0.count }.max() ?? 0
        let mlxTokens = MLXArray(
            tokens
                .map {
                    ($0 + Array(repeating: 0, count: max - $0.count))
                }
                .flatMap { $0 }
        )
        .reshaped(c, max)

        return mlxTokens
    }

    open func step(
        xt: MLXArray, t: MLXArray, tPrev: MLXArray, conditioning: MLXArray, cfgWeight: Float,
        textTime: (MLXArray, MLXArray)?
    ) -> MLXArray {
        let xtUnet = cfgWeight > 1 ? concatenated([xt, xt], axis: 0) : xt
        let tUnet = broadcast(t, to: [xtUnet.count])

        var epsPred = unet(xtUnet, timestep: tUnet, encoderX: conditioning, textTime: textTime)

        if cfgWeight > 1 {
            let (epsText, epsNeg) = epsPred.split()
            epsPred = epsNeg + cfgWeight * (epsText - epsNeg)
        }

        return sampler.step(epsPred: epsPred, xt: xt, t: t, tPrev: tPrev)
    }

    public func detachedDecoder() -> ImageDecoder {
        let autoencoder = self.autoencoder
        func decode(xt: MLXArray) -> MLXArray {
            var x = autoencoder.decode(xt)
            x = clip(x / 2 + 0.5, min: 0, max: 1)
            return x
        }
        return decode(xt:)
    }

    public func decode(xt: MLXArray) -> MLXArray {
        detachedDecoder()(xt)
    }
}

/// Implementation of ``StableDiffusion`` for the `stabilityai/stable-diffusion-2-1-base` model.
open class StableDiffusionBase: StableDiffusion, TextToImageGenerator {

    public init(hub: HubApi, configuration: StableDiffusionConfiguration, dType: DType) throws {
        try super.init(hub: hub, configuration: configuration, dType: dType)
    }

    func conditionText(text: String, imageCount: Int, cfgWeight: Float, negativeText: String?)
        -> MLXArray
    {
        // tokenize the text
        let tokens = tokenize(
            tokenizer: tokenizer, text: text, negativeText: cfgWeight > 1 ? negativeText : nil)

        // compute the features
        var conditioning = textEncoder(tokens).lastHiddenState

        // repeat the conditioning for each of the generated images
        if imageCount > 1 {
            conditioning = repeated(conditioning, count: imageCount, axis: 0)
        }

        return conditioning
    }

    public func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator {
        MLXRandom.seed(parameters.seed)

        let conditioning = conditionText(
            text: parameters.prompt, imageCount: parameters.imageCount,
            cfgWeight: parameters.cfgWeight, negativeText: parameters.negativePrompt)

        let xt = sampler.samplePrior(
            shape: [parameters.imageCount] + parameters.latentSize + [autoencoder.latentChannels],
            dType: dType)

        return DenoiseIterator(
            sd: self, xt: xt, t: sampler.maxTime, conditioning: conditioning,
            steps: parameters.steps, cfgWeight: parameters.cfgWeight)
    }

}

/// Implementation of ``StableDiffusion`` for the `stabilityai/sdxl-turbo` model.
open class StableDiffusionXL: StableDiffusion, TextToImageGenerator, ImageToImageGenerator {

    let textEncoder2: CLIPTextModel
    let tokenizer2: CLIPTokenizer

    public init(hub: HubApi, configuration: StableDiffusionConfiguration, dType: DType) throws {
        let diffusionConfiguration = try loadConfiguration(
            hub: hub, configuration: configuration, key: .diffusionConfig,
            type: DiffusionConfiguration.self)
        let sampler = SimpleEulerAncestralSampler(configuration: diffusionConfiguration)

        self.textEncoder2 = try loadTextEncoder(
            hub: hub, configuration: configuration, configKey: .textEncoderConfig2,
            weightsKey: .textEncoderWeights2, dType: dType)

        self.tokenizer2 = try loadTokenizer(
            hub: hub, configuration: configuration, vocabulary: .tokenizerVocabulary2,
            merges: .tokenizerMerges2)

        try super.init(
            hub: hub, configuration: configuration, dType: dType,
            diffusionConfiguration: diffusionConfiguration, sampler: sampler)
    }

    open override func ensureLoaded() {
        super.ensureLoaded()
        eval(textEncoder2)
    }

    func conditionText(text: String, imageCount: Int, cfgWeight: Float, negativeText: String?) -> (
        MLXArray, MLXArray
    ) {
        let tokens1 = tokenize(
            tokenizer: tokenizer, text: text, negativeText: cfgWeight > 1 ? negativeText : nil)
        let tokens2 = tokenize(
            tokenizer: tokenizer2, text: text, negativeText: cfgWeight > 1 ? negativeText : nil)

        let conditioning1 = textEncoder(tokens1)
        let conditioning2 = textEncoder2(tokens2)
        var conditioning = concatenated(
            [
                conditioning1.hiddenStates.dropLast().last!,
                conditioning2.hiddenStates.dropLast().last!,
            ],
            axis: -1)
        var pooledConditionng = conditioning2.pooledOutput

        if imageCount > 1 {
            conditioning = repeated(conditioning, count: imageCount, axis: 0)
            pooledConditionng = repeated(pooledConditionng, count: imageCount, axis: 0)
        }

        return (conditioning, pooledConditionng)
    }

    public func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator {
        MLXRandom.seed(parameters.seed)

        let (conditioning, pooledConditioning) = conditionText(
            text: parameters.prompt, imageCount: parameters.imageCount,
            cfgWeight: parameters.cfgWeight, negativeText: parameters.negativePrompt)

        let textTime = (
            pooledConditioning,
            repeated(
                MLXArray(converting: [512.0, 512, 0, 0, 512, 512]).reshaped(1, -1),
                count: pooledConditioning.count, axis: 0)
        )

        let xt = sampler.samplePrior(
            shape: [parameters.imageCount] + parameters.latentSize + [autoencoder.latentChannels],
            dType: dType)

        return DenoiseIterator(
            sd: self, xt: xt, t: sampler.maxTime, conditioning: conditioning,
            steps: parameters.steps, cfgWeight: parameters.cfgWeight, textTime: textTime)
    }

    public func generateLatents(image: MLXArray, parameters: EvaluateParameters, strength: Float)
        -> DenoiseIterator
    {
        MLXRandom.seed(parameters.seed)

        // Define the num steps and start step
        let startStep = Float(sampler.maxTime) * strength
        let numSteps = Int(Float(parameters.steps) * strength)

        let (conditioning, pooledConditioning) = conditionText(
            text: parameters.prompt, imageCount: parameters.imageCount,
            cfgWeight: parameters.cfgWeight, negativeText: parameters.negativePrompt)

        let textTime = (
            pooledConditioning,
            repeated(
                MLXArray(converting: [512.0, 512, 0, 0, 512, 512]).reshaped(1, -1),
                count: pooledConditioning.count, axis: 0)
        )

        // Get the latents from the input image and add noise according to the
        // start time.

        var (x0, _) = autoencoder.encode(image[.newAxis])
        x0 = broadcast(x0, to: [parameters.imageCount] + x0.shape.dropFirst())
        let xt = sampler.addNoise(x: x0, t: MLXArray(startStep))

        return DenoiseIterator(
            sd: self, xt: xt, t: sampler.maxTime, conditioning: conditioning, steps: numSteps,
            cfgWeight: parameters.cfgWeight, textTime: textTime)
    }
}
