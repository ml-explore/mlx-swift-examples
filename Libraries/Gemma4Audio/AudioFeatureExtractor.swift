// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXFFT

public class Gemma4AudioFeatureExtractor {
    public let featureSize: Int
    public let samplingRate: Int
    public let paddingValue: Float
    public let frameLength: Int
    public let hopLength: Int
    public let fftLength: Int
    public let minFrequency: Float
    public let maxFrequency: Float
    public let preemphasis: Float
    public let preemphasisHtkFlavor: Bool
    public let fftOverdrive: Bool
    public let dither: Float
    public let inputScaleFactor: Float
    public let melFloor: Float

    public let window: MLXArray
    public let melFilters: MLXArray

    public var perBinMean: MLXArray?
    public var perBinStddev: MLXArray?

    public init(
        featureSize: Int = 128,
        samplingRate: Int = 16000,
        paddingValue: Float = 0.0,
        frameLengthMs: Float = 20.0,
        hopLengthMs: Float = 10.0,
        minFrequency: Float = 0.0,
        maxFrequency: Float = 8000.0,
        preemphasis: Float = 0.0,
        preemphasisHtkFlavor: Bool = true,
        fftOverdrive: Bool = false,
        dither: Float = 0.0,
        inputScaleFactor: Float = 1.0,
        melFloor: Float = 1e-3,
        perBinMean: [Float]? = nil,
        perBinStddev: [Float]? = nil
    ) {
        self.featureSize = featureSize
        self.samplingRate = samplingRate
        self.paddingValue = paddingValue
        self.minFrequency = minFrequency
        self.maxFrequency = maxFrequency
        self.preemphasis = preemphasis
        self.preemphasisHtkFlavor = preemphasisHtkFlavor
        self.fftOverdrive = fftOverdrive
        self.dither = dither
        self.inputScaleFactor = inputScaleFactor
        self.melFloor = melFloor

        self.frameLength = Int(round(Float(samplingRate) * frameLengthMs / 1000.0))
        self.hopLength = Int(round(Float(samplingRate) * hopLengthMs / 1000.0))

        var fftLen = Int(pow(2.0, ceil(log2(Double(self.frameLength)))))
        if fftOverdrive {
            fftLen *= 2
        }
        self.fftLength = fftLen

        // Periodic Hann window
        let n = MLXArray(stride(from: 0, to: self.frameLength, by: 1)).asType(.float32)
        self.window =
            MLXArray(0.5) - MLXArray(0.5)
            * MLX.cos(Float(2.0 * Double.pi) * n / Float(self.frameLength))

        self.melFilters = Gemma4AudioFeatureExtractor.createMelFilterBank(
            numFrequencyBins: self.fftLength / 2 + 1,
            numMelFilters: self.featureSize,
            minFrequency: self.minFrequency,
            maxFrequency: self.maxFrequency,
            samplingRate: self.samplingRate
        )

        if let mean = perBinMean {
            self.perBinMean = MLXArray(mean).reshaped([1, 1, self.featureSize])
        }
        if let stddev = perBinStddev {
            self.perBinStddev = MLXArray(stddev).reshaped([1, 1, self.featureSize])
        }
    }

    private static func hzToMel(_ freq: Float) -> Float {
        return 2595.0 * log10(1.0 + freq / 700.0)
    }

    private static func melToHz(_ mel: Float) -> Float {
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }

    private static func createMelFilterBank(
        numFrequencyBins: Int, numMelFilters: Int, minFrequency: Float, maxFrequency: Float,
        samplingRate: Int
    ) -> MLXArray {
        let melMin = hzToMel(minFrequency)
        let melMax = hzToMel(maxFrequency)

        let step = (melMax - melMin) / Float(numMelFilters + 1)
        var freqPoints = [Float]()
        for i in 0 ..< (numMelFilters + 2) {
            freqPoints.append(melToHz(melMin + Float(i) * step))
        }

        var filterBank = [[Float]](
            repeating: [Float](repeating: 0.0, count: numMelFilters), count: numFrequencyBins)
        let allFreqs = (0 ..< numFrequencyBins).map {
            Float($0) * (Float(samplingRate) / Float(2 * (numFrequencyBins - 1)))
        }

        for i in 0 ..< numMelFilters {
            let lower = freqPoints[i]
            let center = freqPoints[i + 1]
            let upper = freqPoints[i + 2]

            for j in 0 ..< numFrequencyBins {
                let freq = allFreqs[j]
                let rising = (freq - lower) / max(center - lower, 1e-10)
                let falling = (upper - freq) / max(upper - center, 1e-10)
                filterBank[j][i] = max(0.0, min(rising, falling))
            }
        }

        let flatFilterBank = filterBank.flatMap { $0 }
        return MLXArray(flatFilterBank, [numFrequencyBins, numMelFilters])
    }

    private func unfold(array: MLXArray, size: Int, step: Int) -> MLXArray {
        let batchSize = array.dim(0)
        let originalLength = array.dim(1)
        let numFrames = (originalLength - size) / step + 1

        if numFrames <= 0 {
            return MLXArray.zeros([batchSize, 0, size]).asType(array.dtype)
        }

        let strides = [originalLength, step, 1]
        return asStrided(array, [batchSize, numFrames, size], strides: strides)
    }

    public func extractSpectrogram(waveform: MLXArray, attentionMask: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        var x = waveform
        if x.ndim == 1 {
            x = x.expandedDimensions(axes: [0])
        }

        if self.dither > 0.0 {
            let noise = MLXRandom.normal(x.shape).asType(x.dtype)
            x = x + MLXArray(self.dither) * noise
        }

        if self.inputScaleFactor != 1.0 {
            x = x * MLXArray(self.inputScaleFactor)
        }

        let padLeft = self.frameLength / 2
        x = padded(x, widths: [0, [padLeft, 0]])
        var mask = padded(attentionMask, widths: [[padLeft, 0]])

        let frameSizeForUnfold = self.frameLength + 1
        let framesToProcess = unfold(array: x, size: frameSizeForUnfold, step: self.hopLength)

        var frames: MLXArray
        if self.preemphasis > 0.0 {
            if self.preemphasisHtkFlavor {
                let firstInFrame =
                    framesToProcess[MLXEllipsisIndex.ellipsis, 0 ..< 1]
                    * MLXArray(1.0 - self.preemphasis)
                let restInFrame =
                    framesToProcess[MLXEllipsisIndex.ellipsis, 1 ..< (frameSizeForUnfold - 1)]
                    - MLXArray(self.preemphasis)
                    * framesToProcess[MLXEllipsisIndex.ellipsis, 0 ..< (frameSizeForUnfold - 2)]
                frames = MLX.concatenated([firstInFrame, restInFrame], axis: -1)
            } else {
                frames =
                    framesToProcess[MLXEllipsisIndex.ellipsis, 1 ..< frameSizeForUnfold] - MLXArray(
                        self.preemphasis)
                    * framesToProcess[MLXEllipsisIndex.ellipsis, 0 ..< (frameSizeForUnfold - 1)]
            }
        } else {
            frames = framesToProcess[MLXEllipsisIndex.ellipsis, 0 ..< (frameSizeForUnfold - 1)]
        }

        frames = frames * self.window
        let stft = MLXFFT.rfft(frames, n: self.fftLength, axis: -1)

        let magnitudeSpec = MLX.abs(stft)
        let melSpec = MLX.matmul(magnitudeSpec, self.melFilters)
        var logMelSpec = MLX.log(melSpec + MLXArray(self.melFloor))

        if let mean = self.perBinMean {
            logMelSpec = logMelSpec - mean
        }
        if let stddev = self.perBinStddev {
            logMelSpec = logMelSpec / stddev
        }

        let melSpectrogram = logMelSpec.squeezed(axis: 0)
        let numMelFrames = melSpectrogram.dim(0)

        let arange = MLXArray(stride(from: 0, to: numMelFrames, by: 1))
        let frameEndIndices = arange * self.hopLength + (frameSizeForUnfold - 1)
        let specMask = mask[frameEndIndices].asType(.bool)

        return (melSpectrogram, specMask)
    }

    public func callAsFunction(
        rawSpeech: [MLXArray], padding: Bool = true, maxLength: Int? = 480_000,
        padToMultipleOf: Int? = 128
    ) -> (MLXArray, MLXArray) {

        var lengths = rawSpeech.map { $0.size }
        var targetLength = lengths.max() ?? 0

        if let maxL = maxLength {
            targetLength = min(targetLength, maxL)
        }

        if let padTo = padToMultipleOf, targetLength % padTo != 0 {
            targetLength = ((targetLength / padTo) + 1) * padTo
        }

        var paddedSpeech = [MLXArray]()
        var attentionMasks = [MLXArray]()

        for w in rawSpeech {
            var currentW = w
            if currentW.size > targetLength {
                currentW = currentW[0 ..< targetLength]
            }

            var mask = MLXArray.ones([targetLength], type: Int32.self)
            if currentW.size < targetLength {
                let padWidth = targetLength - currentW.size
                mask[currentW.size...] = MLXArray(Int32(0))
                currentW = padded(
                    currentW, widths: [[0, padWidth]], value: MLXArray(self.paddingValue))
            }
            paddedSpeech.append(currentW)
            attentionMasks.append(mask)
        }

        var preparedSpeech = [MLXArray]()
        var preparedSpeechMask = [MLXArray]()

        for i in 0 ..< paddedSpeech.count {
            let speech2d = paddedSpeech[i].reshaped([1, -1])
            let (spec, specMask) = extractSpectrogram(
                waveform: speech2d, attentionMask: attentionMasks[i])
            preparedSpeech.append(spec.asType(.float32))
            preparedSpeechMask.append(specMask)
        }

        // Zero out padded spectrogram positions
        var finalSpeech = [MLXArray]()
        for i in 0 ..< preparedSpeech.count {
            let spec = preparedSpeech[i]
            let m = preparedSpeechMask[i]
            finalSpeech.append(spec * m.expandedDimensions(axes: [-1]))
        }

        return (MLX.stacked(finalSpeech), MLX.stacked(preparedSpeechMask))
    }
}
