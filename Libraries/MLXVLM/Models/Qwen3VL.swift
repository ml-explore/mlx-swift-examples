import CoreImage
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen3_vl

private enum Qwen3VLError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
}

// MARK: - Processor

public final class Qwen3VLProcessor: UserInputProcessor {

    private let config: Qwen3VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Qwen3VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        print("[Qwen3VLProcessor] Normalization: mean=\(config.imageMean) std=\(config.imageStd)")
    }

    private func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
        image
            .toSRGB()
            .resampled(to: resizedSize, method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let processed = images.map { MediaProcessing.apply($0, processing: processing) }

        guard let first = processed.first else {
            throw VLMError.imageProcessingFailure("No image provided")
        }

        let extent = first.extent.size
        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
            height: Int(extent.height),
            width: Int(extent.width),
            factor: config.patchSize * config.mergeSize,
            minPixels: config.size.minPixels,
            maxPixels: config.size.maxPixels)

        let targetSize = CGSize(width: resizedWidth, height: resizedHeight)

        let resampled = processed.map { MediaProcessing.resampleBicubic($0, to: targetSize) }
        
        let normalized = resampled
            .map {
                MediaProcessing.normalize(
                    $0,
                    mean: config.imageMeanTuple,
                    std: config.imageStdTuple)
            }
            .map { MediaProcessing.asMLXArray($0) }
        
        // Debug first image
        if let first = resampled.first {
            let asArray = MediaProcessing.asMLXArray(first)
            print("[Qwen3VL] Before normalization: min=\(asArray.min().item(Float.self)) max=\(asArray.max().item(Float.self)) mean=\(asArray.mean().item(Float.self))")
            print("[Qwen3VL] Before normalization shape: \(asArray.shape) - should be [1, 3, H, W]")
            // Check per-channel stats to detect BGR vs RGB issues
            if asArray.ndim == 4 && asArray.shape[1] == 3 {
                let r = asArray[0, 0, 0..., 0...].mean().item(Float.self)
                let g = asArray[0, 1, 0..., 0...].mean().item(Float.self)
                let b = asArray[0, 2, 0..., 0...].mean().item(Float.self)
                print("[Qwen3VL] Channel means - R:\(r) G:\(g) B:\(b)")
                
                // Sample a few pixel values to verify they're reasonable
                let h = asArray.shape[2]
                let w = asArray.shape[3]
                let midH = h / 2
                let midW = w / 2
                let centerPixel = [
                    asArray[0, 0, midH, midW].item(Float.self),
                    asArray[0, 1, midH, midW].item(Float.self),
                    asArray[0, 2, midH, midW].item(Float.self)
                ]
                print("[Qwen3VL] Center pixel RGB: \(centerPixel) (should be high for white screenshot)")
            }
        }

        return try QwenVL.patchify(
            images: normalized,
            mergeSize: config.mergeSize,
            patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen3VLMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(messages: messages)
        
        if Qwen3VL.debugEnabled {
            print("[Qwen3VL] Processor prepare -> messages: \(messages)")
            print("[Qwen3VL] Processor prepare -> promptTokens count: \(promptTokens.count)")
            if promptTokens.count < 200 {
                print("[Qwen3VL] Processor prepare -> decoded prompt: \(try tokenizer.decode(tokens: promptTokens))")
            }
        }

        if input.images.isEmpty, input.videos.isEmpty {
            // Text-only mode: still need batch dimension [1, seq_len]
            let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray).asType(.int8)
            return LMInput(text: .init(tokens: promptArray, mask: mask))
        }

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imageFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let concatenated = concatenated(imageFrames.map { $0.0 })
            processedImage = .init(pixels: concatenated, frames: imageFrames.map { $0.1 })

            if let frames = processedImage?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens,
                    frames: frames,
                    paddingToken: "<|image_pad|>",
                    mergeSize: config.mergeSize,
                    tokenizer: tokenizer)
            }
        }

        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var accumulatedFrames: [[MLXArray]] = []
            var resizedSize: CGSize = .zero

            for video in input.videos {
                let sequence = try await MediaProcessing.asProcessedSequence(
                    video.asAVAsset(), samplesPerSecond: 2
                ) { frame in
                    let processed = MediaProcessing.apply(frame.frame, processing: input.processing)
                    if resizedSize == .zero {
                        let size = processed.extent.size
                        let (height, width) = try QwenVL.targetSize(
                            height: Int(size.height),
                            width: Int(size.width),
                            factor: config.patchSize * config.mergeSize,
                            minPixels: config.minPixels,
                            maxPixels: config.maxPixels)
                        resizedSize = CGSize(width: width, height: height)
                    }
                    let finalImage = preprocess(image: processed, resizedSize: resizedSize)
                    return VideoFrame(frame: finalImage, timeStamp: frame.timeStamp)
                }
                accumulatedFrames.append(sequence.frames)
            }

            let videoFrames = try accumulatedFrames.map {
                try QwenVL.patchify(
                    images: $0,
                    mergeSize: config.mergeSize,
                    patchSize: config.patchSize,
                    temporalPatchSize: config.temporalPatchSize)
            }

            let concatenated = concatenated(videoFrames.map { $0.0 })
            processedVideo = .init(pixels: concatenated, frames: videoFrames.map { $0.1 })

            if let frames = processedVideo?.frames {
                promptTokens = try QwenVL.replacePaddingTokens(
                    in: promptTokens,
                    frames: frames,
                    paddingToken: "<|video_pad|>",
                    mergeSize: config.mergeSize,
                    tokenizer: tokenizer)
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo)
    }
}

public struct Qwen3VLProcessorConfiguration: Codable, Sendable {

    public struct Size: Codable, Sendable {
        public let maxPixels: Int
        public let minPixels: Int

        enum CodingKeys: String, CodingKey {
            case maxPixels = "max_pixels"
            case minPixels = "min_pixels"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    private let _minPixels: Int?
    private let _maxPixels: Int?
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let imageProcessorType: String
    
    // Provide sensible defaults matching Python mlx-vlm
    public var minPixels: Int { _minPixels ?? 4 * 28 * 28 }      // 3,136
    public var maxPixels: Int { _maxPixels ?? 16384 * 28 * 28 }  // 12,845,056

    public var size: Size { .init(maxPixels: maxPixels, minPixels: minPixels) }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case _minPixels = "min_pixels"
        case _maxPixels = "max_pixels"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case imageProcessorType = "image_processor_type"
    }
}

// MARK: - Model

public final class Qwen3VL: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Qwen3VLLanguage.LanguageModel

    public let config: Qwen3VLConfiguration
    static let debugEnabled = ProcessInfo.processInfo.environment["QWEN3VL_DEBUG"] != nil

    public init(_ config: Qwen3VLConfiguration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen3VLLanguage.LanguageModel(config)
    }

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public func loraLinearLayers() -> LoRALinearLayers {
        languageModel.model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int
    ) throws -> (MLXArray, MLXArray) {
        let imageMask = (inputIds .== MLXArray(imageTokenIndex))
        let videoMask = (inputIds .== MLXArray(videoTokenIndex))
        var specialMask = (imageMask .|| videoMask)
        
        // Count tokens
        let nImageTokens = specialMask.sum().item(Int.self)
        print("[mergeInputIds] Found \(nImageTokens) visual tokens (should match imageFeatures.dim(0)=\(imageFeatures.dim(0)))")
        
        // Expand mask to match embeddings shape [batch, seq, hidden]
        specialMask = expandedDimensions(specialMask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputEmbeds.shape)
        
        // Validate counts
        let nImageFeatures = imageFeatures.dim(0)
        let nImageMaskElements = maskExpanded.sum().item(Int.self)
        let imageFeatureSize = imageFeatures.size
        
        guard nImageMaskElements == imageFeatureSize else {
            throw Qwen3VLError.featureTokenMismatch(expected: nImageTokens, actual: nImageFeatures)
        }
        
        // Implement masked_scatter: flatten, scatter, reshape
        let originalShape = inputEmbeds.shape
        let flattenedEmbeds = inputEmbeds.flattened()
        let flattenedFeatures = imageFeatures.flattened()
        let flattenedMask = maskExpanded.flattened()
        
        // Find positions where mask is true
        let indices = nonZero(flattenedMask.asType(.bool))
        
        // Create result by scattering features into masked positions
        var result = flattenedEmbeds
        if !indices.isEmpty && indices.count == flattenedFeatures.size {
            let indexArray = MLXArray(indices.map { UInt32($0) })
            result[indexArray] = flattenedFeatures
        }
        
        // Reshape back to original shape
        result = result.reshaped(originalShape)
        
        // CRITICAL: Python does image_mask[..., 0] which keeps batch dim
        // specialMask is [batch, seq, hidden], squeeze only last axis to get [batch, seq]
        let visualMask = specialMask.squeezed(axis: -1).asType(.bool)
        return (result, visualMask)
    }

    private func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }

    private func splitFeatures(_ features: MLXArray, sizes: [Int]) -> [MLXArray] {
        guard !sizes.isEmpty else { return [] }
        var offset = 0
        var parts: [MLXArray] = []
        for size in sizes {
            guard size > 0 else { continue }
            let slice = features[offset ..< offset + size, 0...]
            parts.append(slice)
            offset += size
        }
        return parts
    }

    private func combinedFrames(
        imageFrames: [THW]?,
        videoFrames: [THW]?
    ) -> [THW] {
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        return frames
    }

    private func splitSizes(from frames: [THW], mergeSize: Int) -> [Int] {
        frames.map { $0.product / (mergeSize * mergeSize) }
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens
        let inputMask = input.text.mask

        debugLog("prepare invoked with tokens shape \(inputIds.shape) mask shape \(inputMask?.shape ?? [])")

        var pixelValues: MLXArray?
        var imageFrames: [THW]? = nil
        var videoFrames: [THW]? = nil

        let dtype = visionModel.patchEmbed.proj.weight.dtype

        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels.asType(dtype))
            imageFrames = image.frames
        }

        if let video = input.video {
            pixelParts.append(video.pixels.asType(dtype))
            videoFrames = video.frames
        }

        if !pixelParts.isEmpty {
            pixelValues = concatenated(pixelParts)
            if let pixelValues {
                debugLog("pixelValues shape \(pixelValues.shape)")
            }
        }

        // CRITICAL: For text-only, don't pre-compute embeddings! Pass nil and let Model do it.
        // Only create inputEmbeddings when we need to merge with image features.
        var inputEmbeddings: MLXArray? = nil
        var visualMask: MLXArray?
        var deepstackEmbeds: [MLXArray]? = nil

        if let pixelValues, let framesList = combinedFrames(imageFrames: imageFrames, videoFrames: videoFrames).nilIfEmpty {
            // Text+Image mode: pre-compute embeddings for merging
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            let (visionHidden, deepstackOutputs) = visionModel(pixelValues, gridTHW: framesList)
            let mergeSize = config.visionConfiguration.spatialMergeSize
            let splits = splitSizes(from: framesList, mergeSize: mergeSize)

            let featureSlices = splitFeatures(visionHidden, sizes: splits)
            let flattenedFeatures = concatenated(featureSlices).asType(textEmbeds.dtype)

            let (mergedEmbeds, mask) = try mergeInputIdsWithImageFeatures(
                imageFeatures: flattenedFeatures,
                inputEmbeds: textEmbeds,
                inputIds: inputIds,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex)

            inputEmbeddings = mergedEmbeds
            visualMask = mask

            if !deepstackOutputs.isEmpty {
                deepstackEmbeds = deepstackOutputs.map { layerFeatures in
                    let slices = splitFeatures(layerFeatures, sizes: splits)
                    let concatenatedSlices = concatenated(slices).asType(textEmbeds.dtype)
                    return concatenatedSlices
                }
            }
        }
        // For text-only mode, inputEmbeddings stays nil and Model will embed from inputIds

        let typedCache = castCache(cache)

        // Pass BOTH inputIds (for position calculation) and inputEmbeddings (for forward pass)
        let languageOutput = languageModel(
            inputIds,
            cache: typedCache,
            inputEmbeddings: inputEmbeddings,
            mask: nil,
            positionIds: nil,
            visualMask: visualMask,
            deepstackEmbeds: deepstackEmbeds,
            pixelValues: pixelValues,
            imageGridTHW: imageFrames,
            videoGridTHW: videoFrames)

        return .logits(languageOutput)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let typedCache = castCacheOptional(cache)
        
        let offset = cache?.first?.offset ?? 0
        // Debug generation calls
        if offset >= 163 && offset <= 165 {
            print("[Qwen3VL] GENERATION: offset=\(offset) visualMask=nil deepstack=nil (expected during generation)")
        }
        
        let result = languageModel(
            inputs,
            cache: typedCache,
            inputEmbeddings: nil,
            mask: nil,
            positionIds: nil,
            visualMask: nil,
            deepstackEmbeds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        ).logits
        return result
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var adjusted: [String: MLXArray] = [:]
        adjusted.reserveCapacity(weights.count)

        debugLog("sanitize start: incoming tensors = \(weights.count)")
        
        // Debug: check for critical weight keys
        let hasMLP = weights.keys.contains { $0.contains(".mlp.") }
        let hasNorm = weights.keys.contains { $0.contains("model.norm") }
        let hasLayers = weights.keys.contains { $0.contains(".layers.") }
        print("[Qwen3VL] Weight check: hasMLP=\(hasMLP) hasNorm=\(hasNorm) hasLayers=\(hasLayers)")
        
        // Sample some keys
        let sampleKeys = Array(weights.keys.prefix(5))
        print("[Qwen3VL] Sample weight keys: \(sampleKeys)")

        for (key, value) in weights {
            var newKey = key
            if newKey.contains("model.visual") {
                newKey = newKey.replacingOccurrences(of: "model.visual", with: "vision_tower")
            } else if newKey.contains("model.language_model") {
                // CRITICAL: Python remaps to "language_model.model" not just "language_model"
                // This ensures keys like model.language_model.norm.weight -> language_model.model.norm.weight
                newKey = newKey.replacingOccurrences(of: "model.language_model", with: "language_model.model")
            }

            // CRITICAL: Strip "model.lm_head" entirely to "language_model.lm_head"
            if newKey.contains("model.lm_head") {
                newKey = newKey.replacingOccurrences(of: "model.lm_head", with: "language_model.lm_head")
                guard !config.textConfiguration.tieWordEmbeddings else { continue }
                adjusted[newKey] = value
                continue
            } else if newKey.contains("lm_head") && !newKey.contains("language_model") {
                // Handle bare "lm_head" keys
                newKey = newKey.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
                guard !config.textConfiguration.tieWordEmbeddings else { continue }
                adjusted[newKey] = value
                continue
            }

            adjusted[newKey] = value
        }

        debugLog("sanitize remapped keys: \(adjusted.keys.count)")
        
        // Debug: check remapped keys
        let hasMLPAfter = adjusted.keys.contains { $0.contains(".mlp.") }
        let hasNormAfter = adjusted.keys.contains { $0.contains(".norm.weight") }
        let hasLMHead = adjusted.keys.contains { $0.contains("language_model.lm_head") }
        print("[Qwen3VL] After remap: hasMLP=\(hasMLPAfter) hasNorm=\(hasNormAfter) hasLMHead=\(hasLMHead)")
        
        // Debug: Check if layer 0 weights exist
        let layer0Keys = adjusted.keys.filter { $0.hasPrefix("language_model.model.layers.0.") }
        if layer0Keys.count > 0 {
            print("[Qwen3VL] Layer 0 has \(layer0Keys.count) weight keys (should be ~14 for q/k/v/o + norms + mlp)")
        } else {
            print("[Qwen3VL] WARNING: No weights found for layer 0!")
        }

        let sanitized = visionModel.sanitize(weights: adjusted)
        debugLog("sanitize complete: output tensors = \(sanitized.count)")
        return sanitized
    }
}

private extension Array where Element == THW {
    var nilIfEmpty: [THW]? { isEmpty ? nil : self }
}

private extension Qwen3VL {
    func debugLog(_ message: @autoclosure () -> String) {
        guard Qwen3VL.debugEnabled else { return }
        print("[Qwen3VL] \(message())")
    }

    func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }
}

/// Message Generator for Qwen3-VL (images FIRST)
public struct Qwen3VLMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        // For Qwen3-VL, images must come FIRST in the content array
        let imageContent = message.images.map { _ in
            ["type": "image"]
        }
        let textContent = [["type": "text", "text": message.content]]
        let videoContent = message.videos.map { _ in
            ["type": "video"]
        }
        
        return [
            "role": message.role.rawValue,
            "content": imageContent + videoContent + textContent,
        ]
    }
}
