// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXRandom

// port of https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/vae.py

class Attention: Module, UnaryLayer {

    @ModuleInfo(key: "group_norm") public var groupNorm: GroupNorm

    @ModuleInfo(key: "query_proj") public var queryProjection: Linear
    @ModuleInfo(key: "key_proj") public var keyProjection: Linear
    @ModuleInfo(key: "value_proj") public var valueProjection: Linear
    @ModuleInfo(key: "out_proj") public var outProjection: Linear

    init(dimensions: Int, groupCount: Int = 32) {
        self._groupNorm.wrappedValue = GroupNorm(
            groupCount: groupCount, dimensions: dimensions, pytorchCompatible: true)

        self._queryProjection.wrappedValue = Linear(dimensions, dimensions)
        self._keyProjection.wrappedValue = Linear(dimensions, dimensions)
        self._valueProjection.wrappedValue = Linear(dimensions, dimensions)
        self._outProjection.wrappedValue = Linear(dimensions, dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, H, W, C) = x.shape4

        var y = groupNorm(x)

        let queries = queryProjection(y).reshaped(B, H * W, C)
        let keys = keyProjection(y).reshaped(B, H * W, C)
        let values = valueProjection(y).reshaped(B, H * W, C)

        let scale = 1 / sqrt(Float(queries.dim(-1)))
        let scores = (queries * scale).matmul(keys.transposed(0, 2, 1))
        let attention = softmax(scores, axis: -1)

        y = matmul(attention, values).reshaped(B, H, W, C)
        y = outProjection(y)

        return x + y
    }
}

class EncoderDecoderBlock2D: Module, UnaryLayer {

    let resnets: [ResnetBlock2D]
    let downsample: Conv2d?
    let upsample: Conv2d?

    init(
        inputChannels: Int, outputChannels: Int, numLayers: Int = 1, resnetGroups: Int = 32,
        addDownSample: Bool = true, addUpSample: Bool = true
    ) {
        // Add the resnet blocks
        self.resnets = (0 ..< numLayers)
            .map { i in
                ResnetBlock2D(
                    inputChannels: i == 0 ? inputChannels : outputChannels,
                    outputChannels: outputChannels,
                    groupCount: resnetGroups)
            }

        // Add an optional downsampling layer
        if addDownSample {
            self.downsample = Conv2d(
                inputChannels: outputChannels, outputChannels: outputChannels, kernelSize: 3,
                stride: 2, padding: 0)
        } else {
            self.downsample = nil
        }

        // or upsampling layer
        if addUpSample {
            self.upsample = Conv2d(
                inputChannels: outputChannels, outputChannels: outputChannels, kernelSize: 3,
                stride: 1, padding: 1)
        } else {
            self.upsample = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x

        for resnet in resnets {
            x = resnet(x)
        }

        if let downsample {
            x = padded(x, widths: [[0, 0], [0, 1], [0, 1], [0, 0]])
            x = downsample(x)
        }

        if let upsample {
            x = upsample(upsampleNearest(x))
        }

        return x
    }
}

/// Implements the encoder side of the Autoencoder
class VAEncoder: Module, UnaryLayer {

    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var downBlocks: [EncoderDecoderBlock2D]
    @ModuleInfo(key: "mid_blocks") var midBlocks: (ResnetBlock2D, Attention, ResnetBlock2D)
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(
        inputChannels: Int, outputChannels: Int, blockOutChannels: [Int] = [64],
        layersPerBlock: Int = 2, resnetGroups: Int = 32
    ) {
        let channels0 = blockOutChannels[0]

        self._convIn.wrappedValue = Conv2d(
            inputChannels: inputChannels, outputChannels: channels0, kernelSize: 3, stride: 1,
            padding: 1)

        let downblockChannels = [channels0] + blockOutChannels
        self._downBlocks.wrappedValue = zip(downblockChannels, downblockChannels.dropFirst())
            .enumerated()
            .map { (i, pair) in
                let (inChannels, outChannels) = pair
                return EncoderDecoderBlock2D(
                    inputChannels: inChannels, outputChannels: outChannels,
                    numLayers: layersPerBlock, resnetGroups: resnetGroups,
                    addDownSample: i < blockOutChannels.count - 1,
                    addUpSample: false
                )
            }

        let channelsLast = blockOutChannels.last!
        self._midBlocks.wrappedValue = (
            ResnetBlock2D(
                inputChannels: channelsLast,
                outputChannels: channelsLast,
                groupCount: resnetGroups
            ),
            Attention(dimensions: channelsLast, groupCount: resnetGroups),
            ResnetBlock2D(
                inputChannels: channelsLast,
                outputChannels: channelsLast,
                groupCount: resnetGroups
            )
        )

        self._convNormOut.wrappedValue = GroupNorm(
            groupCount: resnetGroups, dimensions: channelsLast, pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(
            inputChannels: channelsLast, outputChannels: outputChannels,
            kernelSize: 3,
            padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = convIn(x)

        for l in downBlocks {
            x = l(x)
        }

        x = midBlocks.0(x)
        x = midBlocks.1(x)
        x = midBlocks.2(x)

        x = convNormOut(x)
        x = silu(x)
        x = convOut(x)

        return x
    }
}

/// Implements the decoder side of the Autoencoder
class VADecoder: Module, UnaryLayer {

    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_blocks") var midBlocks: (ResnetBlock2D, Attention, ResnetBlock2D)
    @ModuleInfo(key: "up_blocks") var upBlocks: [EncoderDecoderBlock2D]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(
        inputChannels: Int, outputChannels: Int, blockOutChannels: [Int] = [64],
        layersPerBlock: Int = 2, resnetGroups: Int = 32
    ) {
        let channels0 = blockOutChannels[0]
        let channelsLast = blockOutChannels.last!

        self._convIn.wrappedValue = Conv2d(
            inputChannels: inputChannels, outputChannels: channelsLast, kernelSize: 3, stride: 1,
            padding: 1)

        self._midBlocks.wrappedValue = (
            ResnetBlock2D(
                inputChannels: channelsLast,
                outputChannels: channelsLast,
                groupCount: resnetGroups
            ),
            Attention(dimensions: channelsLast, groupCount: resnetGroups),
            ResnetBlock2D(
                inputChannels: channelsLast,
                outputChannels: channelsLast,
                groupCount: resnetGroups
            )
        )

        let channels = [channelsLast] + blockOutChannels.reversed()
        self._upBlocks.wrappedValue = zip(channels, channels.dropFirst())
            .enumerated()
            .map { (i, pair) in
                let (inChannels, outChannels) = pair
                return EncoderDecoderBlock2D(
                    inputChannels: inChannels,
                    outputChannels: outChannels,
                    numLayers: layersPerBlock,
                    resnetGroups: resnetGroups,
                    addDownSample: false,
                    addUpSample: i < blockOutChannels.count - 1
                )
            }

        self._convNormOut.wrappedValue = GroupNorm(
            groupCount: resnetGroups, dimensions: channels0, pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(
            inputChannels: channels0, outputChannels: outputChannels,
            kernelSize: 3,
            padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = convIn(x)

        x = midBlocks.0(x)
        x = midBlocks.1(x)
        x = midBlocks.2(x)

        for l in upBlocks {
            x = l(x)
        }

        x = convNormOut(x)
        x = silu(x)
        x = convOut(x)

        return x
    }
}

/// The autoencoder that allows us to perform diffusion in the latent space
class Autoencoder: Module {

    let latentChannels: Int
    let scalingFactor: Float
    let encoder: VAEncoder
    let decoder: VADecoder

    @ModuleInfo(key: "quant_proj") public var quantProjection: Linear
    @ModuleInfo(key: "post_quant_proj") public var postQuantProjection: Linear

    init(configuration: AutoencoderConfiguration) {
        self.latentChannels = configuration.latentChannelsIn
        self.scalingFactor = configuration.scalingFactor
        self.encoder = VAEncoder(
            inputChannels: configuration.inputChannels,
            outputChannels: configuration.latentChannelsOut,
            blockOutChannels: configuration.blockOutChannels,
            layersPerBlock: configuration.layersPerBlock,
            resnetGroups: configuration.normNumGroups)
        self.decoder = VADecoder(
            inputChannels: configuration.latentChannelsIn,
            outputChannels: configuration.outputChannels,
            blockOutChannels: configuration.blockOutChannels,
            layersPerBlock: configuration.layersPerBlock + 1,
            resnetGroups: configuration.normNumGroups)

        self._quantProjection.wrappedValue = Linear(
            configuration.latentChannelsIn, configuration.latentChannelsOut)
        self._postQuantProjection.wrappedValue = Linear(
            configuration.latentChannelsIn, configuration.latentChannelsIn)
    }

    func decode(_ z: MLXArray) -> MLXArray {
        let z = z / scalingFactor
        return decoder(postQuantProjection(z))
    }

    func encode(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var x = encoder(x)
        x = quantProjection(x)
        var (mean, logvar) = x.split(axis: -1)
        mean = mean * scalingFactor
        logvar = logvar + 2 * log(scalingFactor)

        return (mean, logvar)
    }

    struct Result {
        let xHat: MLXArray
        let z: MLXArray
        let mean: MLXArray
        let logvar: MLXArray
    }

    func callAsFunction(_ x: MLXArray, key: MLXArray? = nil) -> Result {
        let (mean, logvar) = encode(x)
        let z = MLXRandom.normal(mean.shape, key: key) * exp(0.5 * logvar) + mean
        let xHat = decode(z)

        return Result(xHat: xHat, z: z, mean: mean, logvar: logvar)
    }
}
