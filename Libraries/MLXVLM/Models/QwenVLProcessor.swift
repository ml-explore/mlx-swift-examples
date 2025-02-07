import CoreImage
import Foundation
import MLX
import MLXLMCommon
import Tokenizers

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

    func applyChatTemplate(messages: [Message], tokenizer: any Tokenizer) throws -> [Int]
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

// Base processor class
public class QwenVLProcessor<Config: QwenVLProcessorConfiguration>: UserInputProcessor {
    private let config: Config
    private let tokenizer: any Tokenizer

    public init(_ config: Config, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func targetSize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int)
        throws -> (Int, Int)
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
            hBar = Int(floor(Float(height) * beta / Float(factor))) * factor
            wBar = Int(floor(Float(width) * beta / Float(factor))) * factor
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
        let mod = patches.dim(0) % config.temporalPatchSize
        if mod != 0 {
            let lastPatch = patches[-1, .ellipsis]
            let lastPatchRepeated = tiled(
                lastPatch, repetitions: [config.temporalPatchSize - mod, 1, 1, 1])
            patches = concatenated([patches, lastPatchRepeated])
        }
        let channel = patches.dim(1)
        let gridT = patches.dim(0) / self.config.temporalPatchSize
        let gridH = resizedHeight / self.config.patchSize
        let gridW = resizedWidth / self.config.patchSize

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

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = input.prompt.asMessages()
        var promptTokens = try config.applyChatTemplate(messages: messages, tokenizer: tokenizer)

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
