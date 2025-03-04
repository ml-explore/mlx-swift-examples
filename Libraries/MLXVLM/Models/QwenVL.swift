import CoreImage
import Foundation
import Hub
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Common Utilities

/// Rotates half the hidden dims of the input
public func rotateHalf(_ x: MLXArray) -> MLXArray {
    let index = x.dim(-1) / 2
    let x1 = x[.ellipsis, 0 ..< index]
    let x2 = x[.ellipsis, index...]
    return concatenated([-x2, x1], axis: -1)
}

// MARK: - Language Model Components

public enum QwenVLLanguage {
    /// Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors
    public static func applyMultimodalRotaryPositionEmbedding(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray,
        positionIds: MLXArray, mropeSection: [Int]
    ) -> (MLXArray, MLXArray) {
        var cos = cos[positionIds]
        var sin = sin[positionIds]

        cos =
            concatenated(
                // [m[i % 3] for i, m in enumerate(mx.split(cos, mrope_section, axis=-1))]
                split(cos, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
                axis: -1
            )[0..., .newAxis, 0..., 0...]

        sin =
            concatenated(
                split(sin, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
                axis: -1
            )[0..., .newAxis, 0..., 0...]

        // Apply rotary embedding
        let qEmbed = (q * cos) + (rotateHalf(q) * sin)
        let kEmbed = (k * cos) + (rotateHalf(k) * sin)
        return (qEmbed, kEmbed)
    }

    public class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let mropeSection: [Int]

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        public init(hiddenSize: Int, attentionHeads: Int, kvHeads: Int, ropeTheta: Float, ropeTraditional: Bool, ropeScaling: [String: StringOrNumber]?) {
            self.heads = attentionHeads
            self.kvHeads = kvHeads
            self.headDim = hiddenSize / attentionHeads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(hiddenSize, heads * headDim, bias: true)
            self._wk.wrappedValue = Linear(hiddenSize, kvHeads * headDim, bias: true)
            self._wv.wrappedValue = Linear(hiddenSize, kvHeads * headDim, bias: true)
            self._wo.wrappedValue = Linear(heads * headDim, hiddenSize, bias: false)

            if let v = ropeScaling?["mrope_section"], let array = v.asInts() {
                // mrope_section = np.cumsum(mrope_section * 2)[:-1].tolist()
                self.mropeSection = sequence(state: (0, array.makeIterator())) { state in
                    if let v = state.1.next() {
                        // note the *2
                        state.0 += v * 2
                        return state.0
                    } else {
                        return nil
                    }
                }.dropLast()
            } else {
                fatalError("rope_scaling['mrope_section'] must be an array of integers")
            }

            self._rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim, traditional: ropeTraditional, base: ropeTheta)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            // prepare the queries, keys and values for the attention computation
            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            let mask = mask?[0..., 0 ..< keys.dim(-2)]

            queries = rotaryEmbedding(queries, offset: offset)
            keys = rotaryEmbedding(keys, offset: offset)

            if let cache {
                (keys, values) = cache.update(keys: keys, values: values)
            }

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    public class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }
}

// MARK: - Vision Model Components

public enum QwenVLVision {
    public static func applyMultimodalRotaryPositionEmbedding(
        _ tensor: MLXArray, freqs: MLXArray
    ) -> MLXArray {
        var cos = cos(freqs)
        var sin = sin(freqs)

        cos = expandedDimensions(cos, axis: 1)
        cos = tiled(cos, repetitions: [1, 1, 2])
        cos = expandedDimensions(cos, axis: 0)

        sin = expandedDimensions(sin, axis: 1)
        sin = tiled(sin, repetitions: [1, 1, 2])
        sin = expandedDimensions(sin, axis: 0)

        let output = (tensor * cos) + (rotateHalf(tensor) * sin)
        return output.asType(tensor.dtype)
    }

    public class VisionRotaryEmbedding: Module {
        let dimensions: Int
        let theta: Float

        public init(dimensions: Int, theta: Float = 10000.0) {
            self.dimensions = dimensions
            self.theta = theta
        }

        public func callAsFunction(_ sequenceLength: Int) -> MLXArray {
            let p = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / dimensions
            let inverseFreq = 1.0 / pow(theta, p)
            let seq = MLXArray(0 ..< sequenceLength).asType(inverseFreq.dtype)
            return outer(seq, inverseFreq)
        }
    }

    public class PatchEmbed: Module {
        @ModuleInfo var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let embedDimensions: Int

        public init(patchSize: Int, temporalPatchSize: Int, inChannels: Int, embedDimensions: Int) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.embedDimensions = embedDimensions

            let kernelSize = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            self._proj.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: embedDimensions,
                kernelSize: kernelSize,
                stride: kernelSize,
                bias: false
            )
        }

        public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            var hiddenStates = hiddenStates.reshaped(
                -1, inChannels, temporalPatchSize, patchSize, patchSize
            ).movedAxis(source: 1, destination: 4)

            hiddenStates = proj(hiddenStates)
            hiddenStates = hiddenStates.reshaped(-1, embedDimensions)
            return hiddenStates
        }
    }

    class PatchMerger: Module {
        let hiddenSize: Int

        @ModuleInfo(key: "ln_q") var layerNormQ: RMSNorm
        @ModuleInfo var mlp: (Linear, GELU, Linear)

        init(dimensions: Int, contextDimensions: Int, spatialMergeSize: Int = 2) {
            self.hiddenSize = contextDimensions * (spatialMergeSize * spatialMergeSize)
            self._layerNormQ.wrappedValue = RMSNorm(dimensions: contextDimensions, eps: 1e-6)
            self.mlp = (
                Linear(hiddenSize, hiddenSize),
                GELU(),
                Linear(hiddenSize, dimensions)
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var x = layerNormQ(x).reshaped(-1, hiddenSize)
            x = mlp.0(x)
            x = mlp.1(x)
            x = mlp.2(x)
            return x
        }
    }

    class Attention: Module {
        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        public init(dims: Int, numHeads: Int) {
            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qkv.wrappedValue = Linear(dims, 3 * dims, bias: true)
            self._proj.wrappedValue = Linear(dims, dims)
        }

        public func callAsFunction(
            _ x: MLXArray, frames: [THW], rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)
            let B = frames[0].t
            let L = sequenceLength / B

            let qkv = qkv(x)
            let s = split(qkv, parts: 3, axis: -1)
            var (q, k, v) = (s[0], s[1], s[2])

            q = q.reshaped(sequenceLength, numHeads, -1)
            k = k.reshaped(sequenceLength, numHeads, -1)
            v = v.reshaped(sequenceLength, numHeads, -1)

            q = QwenVLVision.applyMultimodalRotaryPositionEmbedding(q, freqs: rotaryPositionEmbedding)
            k = QwenVLVision.applyMultimodalRotaryPositionEmbedding(k, freqs: rotaryPositionEmbedding)

            q = q.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            k = k.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            v = v.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: nil
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)

            return proj(output)
        }
    }
}

// MARK: - Model Configuration Base Classes

/// Base protocol for Qwen VL configuration
public protocol QwenVLBaseConfiguration: Codable, Sendable {
    var vocabularySize: Int { get }
    var imageTokenId: Int { get }
    var videoTokenId: Int { get }
    var hiddenSize: Int { get }
}

/// Base protocol for text configuration
public protocol QwenVLTextConfigurable: Codable, Sendable {
    var hiddenSize: Int { get }
    var hiddenLayers: Int { get }
    var intermediateSize: Int { get }
    var attentionHeads: Int { get }
    var rmsNormEps: Float { get }
    var vocabularySize: Int { get }
    var kvHeads: Int { get }
    var ropeTheta: Float { get }
    var ropeTraditional: Bool { get }
    var ropeScaling: [String: StringOrNumber]? { get }
    var tieWordEmbeddings: Bool { get }
}

/// Base protocol for vision configuration
public protocol QwenVLVisionConfigurable: Codable, Sendable {
    var patchSize: Int { get }
    var inChannels: Int { get }
    var temporalPatchSize: Int { get }
    var spatialMergeSize: Int { get }
}

// MARK: - Common Processor Configuration

/// Configuration for the Qwen VL processor
public protocol QwenVLProcessorConfiguration: Codable, Sendable {
    var imageMean: [CGFloat] { get }
    var imageStd: [CGFloat] { get }
    var maxPixels: Int { get }
    var minPixels: Int { get }
    var mergeSize: Int { get }
    var patchSize: Int { get }
    var temporalPatchSize: Int { get }

    var imageMeanTuple: (CGFloat, CGFloat, CGFloat) { get }
    var imageStdTuple: (CGFloat, CGFloat, CGFloat) { get }
}

// Default implementation for common properties
extension QwenVLProcessorConfiguration {
    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }
}

// MARK: - Common VLM Model Functions

public extension VLMModel where Self: Module, Self: KVCacheDimensionProvider {
    /// Common implementation for merging input IDs with image features
    func mergeInputIdsWithImageFeatures(
        inputIds: MLXArray,
        inputEmbeds: MLXArray,
        imageFeatures: MLXArray,
        imageTokenId: Int,
        videoTokenId: Int
    ) -> MLXArray {
        var imageIndices = [Int]()
        for (i, v) in inputIds.asArray(Int.self).enumerated() {
            if v == imageTokenId || v == videoTokenId {
                imageIndices.append(i)
            }
        }

        // Make sure shapes match before assignment
        var result = inputEmbeds
        if result.ndim == 2 {
            result = result[.newAxis, 0..., 0...]
        }

        if imageFeatures.ndim == 2 {
            let reshapedFeatures = imageFeatures[.newAxis, 0..., 0...]
            result[0..., MLXArray(imageIndices), 0...] = reshapedFeatures
        } else {
            result[0..., MLXArray(imageIndices), 0...] = imageFeatures
        }

        return result
    }

    /// Helper method to determine if an array is in MLX weight format
    func isMLXWeight(_ array: MLXArray) -> Bool {
        if array.ndim != 4, array.ndim != 5 {
            return false
        }

        if array.dim(-1) == 3 {
            return true
        }

        let (outChannels, kH, kW) = (array.dim(1), array.dim(2), array.dim(3))
        return outChannels >= kH && outChannels >= kW && kH == kW
    }

    /// Helper method to sanitize PyTorch weights for MLX
    func sanitizeVisionWeights(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = [String: MLXArray]()

        for (k, v) in weights {
            if k.contains("position_id") {
                // Remove unused position_ids
                continue
            } else if k.contains("patch_embed.proj.weight") {
                // PyTorch conv2d weight tensors have shape:
                //   [B, out_channels, in_channels, kH, KW]
                // MLX conv2d expects the weight be of shape:
                //   [B, out_channels, kH, KW, in_channels]
                if isMLXWeight(v) {
                    sanitizedWeights[k] = v
                } else {
                    sanitizedWeights[k] = v.transposed(0, 2, 3, 4, 1)
                }
            } else {
                sanitizedWeights[k] = v
            }
        }

        return sanitizedWeights
    }
}

// MARK: - Common Processor Implementation

/// Base implementation for Qwen VL processors
public class QwenVLProcessor<Config: QwenVLProcessorConfiguration>: UserInputProcessor {

    public let config: Config
    public let tokenizer: any Tokenizer

    public init(config: Config, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    func targetSize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int)
    throws
    -> (Int, Int)
    {
        if height < factor {
            throw VLMError.imageProcessingFailure(
                "height: \(height) must be larger than factor: \(factor)")
        }
        if width < factor {
            throw VLMError.imageProcessingFailure(
                "width: \(width) must be larger than factor: \(factor)")
        }
        if max(height, width) / min(height, width) > 200 {
            throw VLMError.imageProcessingFailure(
                "absolute aspect ratio must be smaller than 200: \(width)x\(height)")
        }

        var hBar = max(factor, Int(round(Float(height) / Float(factor))) * factor)
        var wBar = max(factor, Int(round(Float(width) / Float(factor))) * factor)

        if hBar * wBar > maxPixels {
            let beta = sqrt(Float(height * width) / Float(maxPixels))
            hBar = Int(floor(Float(height) / beta / Float(factor))) * factor
            wBar = Int(floor(Float(width) / beta / Float(factor))) * factor
        } else if hBar * wBar < minPixels {
            let beta = sqrt(Float(minPixels) / Float(height * width))
            hBar = Int(ceil(Float(height) * beta / Float(factor))) * factor
            wBar = Int(ceil(Float(width) * beta / Float(factor))) * factor
        }
        return (hBar, wBar)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        // first apply the user requested resizing, etc. if any
        let images = images.map { MediaProcessing.apply($0, processing: processing) }

        // image_processing_qwen2_vl._preprocess
        let size = images[0].extent.size

        let (resizedHeight, resizedWidth) = try targetSize(
            height: Int(size.height), width: Int(size.width),
            factor: config.patchSize * config.mergeSize,
            minPixels: config.minPixels, maxPixels: config.maxPixels)

        // Apply the calculated dimensions
        let resizedSize = CGSize(width: resizedWidth, height: resizedHeight)

        let processedImages =
        try images
            .map {
                MediaProcessing.inSRGBToneCurveSpace($0)
            }
            .map {
                MediaProcessing.resampleBicubic($0, to: resizedSize)
            }
            .map {
                MediaProcessing.normalize(
                    $0, mean: config.imageMeanTuple, std: config.imageStdTuple)
            }
            .map {
                MediaProcessing.asMLXArray($0)
            }

        var patches = concatenated(processedImages)

        // Handle temporal dimension
        let mod = patches.dim(0) % config.temporalPatchSize
        if mod != 0 {
            let lastPatch = patches[-1, .ellipsis]
            let lastPatchRepeated = tiled(
                lastPatch, repetitions: [config.temporalPatchSize - mod, 1, 1, 1])
            patches = concatenated([patches, lastPatchRepeated])
        }

        // Calculate grid dimensions
        let channel = patches.dim(1)
        let gridT = patches.dim(0) / self.config.temporalPatchSize
        let gridH = patches.dim(2) / self.config.patchSize
        let gridW = patches.dim(3) / self.config.patchSize

        patches = patches.reshaped(
            gridT,
            config.temporalPatchSize,
            channel,
            gridH / config.mergeSize,
            config.mergeSize,
            config.patchSize,
            gridW / config.mergeSize,
            config.mergeSize,
            config.patchSize
        )
        patches = patches.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)

        let flattenedPatches = patches.reshaped(
            gridT * gridH * gridW,
            channel * config.temporalPatchSize * config.patchSize * config.patchSize
        )

        return (flattenedPatches, .init(gridT, gridH, gridW))
    }

    public func prepare(input: MLXLMCommon.UserInput) async throws -> MLXLMCommon.LMInput {
        let messages = input.prompt.asMessages()
        var promptTokens = try tokenizer.applyChatTemplate(messages: messages)

        // Text-only input
        if input.images.isEmpty, input.videos.isEmpty {
            return LMInput(tokens: MLXArray(promptTokens))
        }

        // Process images if any
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated, frames: imagePixelsAndFrames.map { $0.1 })
            if let imageFrames = processedImage?.frames {
                promptTokens = try replacePaddingTokens(
                    in: promptTokens, frames: imageFrames, paddingToken: "<|image_pad|>")
            }
        }

        // Process videos if any
        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var videosAsImageSequences = [[CIImage]]()
            for video in input.videos {
                if let imageSequence = try? await MediaProcessing.asCIImageSequence(
                    video.asAVAsset(), samplesPerSecond: 2)
                {
                    videosAsImageSequences.append(imageSequence)
                }
            }
            let videoPixelsAndFrames = try videosAsImageSequences.map {
                try preprocess(images: $0, processing: input.processing)
            }
            let videoPixelsConcatenated = concatenated(videoPixelsAndFrames.map { $0.0 })
            processedVideo = LMInput.ProcessedVideo(
                pixels: videoPixelsConcatenated, frames: videoPixelsAndFrames.map { $0.1 })
            if let videoFrames = processedVideo?.frames {
                promptTokens = try replacePaddingTokens(
                    in: promptTokens, frames: videoFrames, paddingToken: "<|video_pad|>")
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo)
    }

    func replacePaddingTokens(in promptTokens: [Int], frames: [THW], paddingToken: String)
    throws -> [Int]
    {
        // Replace single padding token with correct number for each image or video frame
        let placeholderTokens = try tokenizer.encode(
            text: "<|vision_start|>\(paddingToken)<|vision_end|>")
        let placeholderRanges = promptTokens.ranges(of: placeholderTokens)
        guard placeholderRanges.count == frames.count else {
            throw VLMError.processing(
                "Number of placeholder tokens does not match number of frames")
        }
        let mergeLength = config.mergeSize * config.mergeSize
        let replacementSequences = try frames.map { frame in
            let paddingCount = frame.product / mergeLength
            return try tokenizer.encode(
                text:
                    "<|vision_start|>\(Array(repeating: paddingToken, count: paddingCount).joined())<|vision_end|>"
            )
        }
        // Build the final array
        var result: [Int] = []
        var currentIndex = promptTokens.startIndex
        for (range, replacement) in zip(placeholderRanges, replacementSequences) {
            // Add tokens before the placeholder
            result.append(contentsOf: promptTokens[currentIndex ..< range.lowerBound])
            // Add replacement sequence
            result.append(contentsOf: replacement)
            currentIndex = range.upperBound
        }
        // Add any remaining tokens after the last replacement
        if currentIndex < promptTokens.endIndex {
            result.append(contentsOf: promptTokens[currentIndex...])
        }
        return result
    }
}
