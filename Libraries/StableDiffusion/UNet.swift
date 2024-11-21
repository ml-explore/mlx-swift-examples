// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// port of https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/unet.py

func upsampleNearest(_ x: MLXArray, scale: Int = 2) -> MLXArray {
    precondition(x.ndim == 4)
    let (B, H, W, C) = x.shape4
    var x = broadcast(
        x[0..., 0..., .newAxis, 0..., .newAxis, 0...], to: [B, H, scale, W, scale, C])
    x = x.reshaped(B, H * scale, W * scale, C)
    return x
}

class TimestepEmbedding: Module, UnaryLayer {

    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputChannels: Int, timeEmbedDimensions: Int) {
        self._linear1.wrappedValue = Linear(inputChannels, timeEmbedDimensions)
        self._linear2.wrappedValue = Linear(timeEmbedDimensions, timeEmbedDimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = linear1(x)
        x = silu(x)
        x = linear2(x)

        return x
    }
}

class TransformerBlock: Module {

    let norm1: LayerNorm
    let attn1: MultiHeadAttention

    let norm2: LayerNorm
    let attn2: MultiHeadAttention

    let norm3: LayerNorm
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    @ModuleInfo var linear3: Linear

    init(
        modelDimensions: Int, numHeads: Int, hiddenDimensions: Int? = nil,
        memoryDimensions: Int? = nil
    ) {
        norm1 = LayerNorm(dimensions: modelDimensions)
        attn1 = MultiHeadAttention(dimensions: modelDimensions, numHeads: numHeads)

        // we want to self.attn1.out_proj.bias = mx.zeros(model_dims) turn enable the
        // bias in one of the four Linears attached to attn1.  Since bias is nil we can't
        // update it so just replace the layer.
        attn1.update(
            modules: ModuleChildren(
                values: ["out_proj": .value(Linear(modelDimensions, modelDimensions, bias: true))]))

        let memoryDimensions = memoryDimensions ?? modelDimensions
        self.norm2 = LayerNorm(dimensions: modelDimensions)
        self.attn2 = MultiHeadAttention(
            dimensions: modelDimensions, numHeads: numHeads, keyInputDimensions: memoryDimensions)
        attn2.update(
            modules: ModuleChildren(
                values: ["out_proj": .value(Linear(modelDimensions, modelDimensions, bias: true))]))

        let hiddenDimensions = hiddenDimensions ?? (4 * modelDimensions)
        self.norm3 = LayerNorm(dimensions: modelDimensions)
        self.linear1 = Linear(modelDimensions, hiddenDimensions)
        self.linear2 = Linear(modelDimensions, hiddenDimensions)
        self.linear3 = Linear(hiddenDimensions, modelDimensions)
    }

    func callAsFunction(
        _ x: MLXArray, memory: MLXArray, attentionMask: MLXArray?, memoryMask: MLXArray?
    ) -> MLXArray {
        var x = x

        // self attention
        var y = norm1(x)
        y = attn1(y, keys: y, values: y, mask: attentionMask)
        x = x + y

        // cross attention
        y = norm2(x)
        y = attn2(y, keys: memory, values: memory, mask: memoryMask)
        x = x + y

        // FFN
        y = norm3(x)
        let ya = linear1(y)
        let yb = linear2(y)
        y = ya * gelu(yb)
        y = linear3(y)
        x = x + y

        return x
    }
}

/// A transformer model for inputs with 2 spatial dimensions
class Transformer2D: Module {

    let norm: GroupNorm
    @ModuleInfo(key: "proj_in") var projectIn: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [TransformerBlock]
    @ModuleInfo(key: "proj_out") var projectOut: Linear

    init(
        inputChannels: Int, modelDimensions: Int, encoderDimensions: Int, numHeads: Int,
        numLayers: Int, groupCount: Int = 32
    ) {
        self.norm = GroupNorm(
            groupCount: groupCount, dimensions: inputChannels, pytorchCompatible: true)
        self._projectIn.wrappedValue = Linear(inputChannels, modelDimensions)
        self._transformerBlocks.wrappedValue = (0 ..< numLayers)
            .map { _ in
                TransformerBlock(
                    modelDimensions: modelDimensions, numHeads: numHeads,
                    memoryDimensions: encoderDimensions)
            }
        self._projectOut.wrappedValue = Linear(modelDimensions, inputChannels)
    }

    func callAsFunction(
        _ x: MLXArray, encoderX: MLXArray, attentionMask: MLXArray?, encoderAttentionMask: MLXArray?
    ) -> MLXArray {
        let inputX = x
        let dtype = x.dtype
        var x = x

        // Perform the input norm and projection
        let (B, H, W, C) = x.shape4
        x = norm(x).reshaped(B, -1, C)
        x = projectIn(x)

        // apply the transformer
        for block in transformerBlocks {
            x = block(
                x, memory: encoderX, attentionMask: attentionMask, memoryMask: encoderAttentionMask)
        }

        // apply the output projection and reshape
        x = projectOut(x)
        x = x.reshaped(B, H, W, C)

        return x + inputX
    }
}

class ResnetBlock2D: Module {

    let norm1: GroupNorm
    let conv1: Conv2d

    @ModuleInfo(key: "time_emb_proj") var timeEmbedProjection: Linear?

    let norm2: GroupNorm
    let conv2: Conv2d

    @ModuleInfo(key: "conv_shortcut") var convolutionShortcut: Linear?

    init(
        inputChannels: Int, outputChannels: Int? = nil, groupCount: Int = 32,
        timeEmbedChannels: Int? = nil
    ) {
        let outputChannels = outputChannels ?? inputChannels

        self.norm1 = GroupNorm(
            groupCount: groupCount, dimensions: inputChannels, pytorchCompatible: true)
        self.conv1 = Conv2d(
            inputChannels: inputChannels, outputChannels: outputChannels,
            kernelSize: 3, stride: 1, padding: 1)

        if let timeEmbedChannels {
            self._timeEmbedProjection.wrappedValue = Linear(timeEmbedChannels, outputChannels)
        }

        self.norm2 = GroupNorm(
            groupCount: groupCount, dimensions: outputChannels, pytorchCompatible: true)
        self.conv2 = Conv2d(
            inputChannels: outputChannels, outputChannels: outputChannels,
            kernelSize: 3, stride: 1, padding: 1)

        if inputChannels != outputChannels {
            self._convolutionShortcut.wrappedValue = Linear(inputChannels, outputChannels)
        }
    }

    func callAsFunction(_ x: MLXArray, timeEmbedding: MLXArray? = nil) -> MLXArray {
        let dtype = x.dtype

        var y = norm1(x)
        y = silu(y)
        y = conv1(y)

        if var timeEmbedding, let timeEmbedProjection {
            timeEmbedding = timeEmbedProjection(silu(timeEmbedding))
            y = y + timeEmbedding[0..., .newAxis, .newAxis, 0...]
        }

        y = norm2(y)
        y = silu(y)
        y = conv2(y)

        if let convolutionShortcut {
            return y + convolutionShortcut(x)
        } else {
            return y + x
        }
    }
}

class UNetBlock2D: Module {

    let resnets: [ResnetBlock2D]
    let attentions: [Transformer2D]?
    let downsample: Conv2d?
    let upsample: Conv2d?

    init(
        inputChannels: Int, outputChannels: Int, timeEmbedChannels: Int,
        previousOutChannels: Int? = nil, numLayers: Int = 1, transformerLayersPerBlock: Int = 1,
        numHeads: Int = 8, crossAttentionDimension: Int = 1280, resnetGroups: Int = 32,
        addDownSample: Bool = true, addUpSample: Bool = true, addCrossAttention: Bool = true
    ) {

        // Prepare the inputChannelsArray for the resnets
        let inputChannelsArray: [Int]
        if let previousOutChannels {
            let inputChannelsBuild =
                [previousOutChannels] + Array(repeating: outputChannels, count: numLayers - 1)
            let resChannelsArray =
                Array(repeating: outputChannels, count: numLayers - 1) + [inputChannels]
            inputChannelsArray = zip(inputChannelsBuild, resChannelsArray).map { $0.0 + $0.1 }
        } else {
            inputChannelsArray =
                [inputChannels] + Array(repeating: outputChannels, count: numLayers - 1)
        }

        // Add resnet blocks that also process the time embedding
        self.resnets =
            inputChannelsArray
            .map { ic in
                ResnetBlock2D(
                    inputChannels: ic, outputChannels: outputChannels, groupCount: resnetGroups,
                    timeEmbedChannels: timeEmbedChannels)
            }

        // Add optional cross attention layers
        if addCrossAttention {
            self.attentions = (0 ..< numLayers)
                .map { _ in
                    Transformer2D(
                        inputChannels: outputChannels, modelDimensions: outputChannels,
                        encoderDimensions: crossAttentionDimension, numHeads: numHeads,
                        numLayers: transformerLayersPerBlock)
                }
        } else {
            self.attentions = nil
        }

        // Add an optional downsampling layer
        if addDownSample {
            self.downsample = Conv2d(
                inputChannels: outputChannels, outputChannels: outputChannels, kernelSize: 3,
                stride: 2, padding: 1)
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

    func callAsFunction(
        _ x: MLXArray, encoderX: MLXArray, timeEmbedding: MLXArray? = nil,
        attentionMask: MLXArray? = nil, encoderAttentionMask: MLXArray? = nil,
        residualHiddenStates: [MLXArray]? = nil
    ) -> (MLXArray, [MLXArray], [MLXArray]) {
        var x = x
        var outputStates = [MLXArray]()
        var residualHiddenStates = residualHiddenStates

        for i in 0 ..< resnets.count {
            if residualHiddenStates != nil {
                x = concatenated([x, residualHiddenStates!.removeLast()], axis: -1)
            }

            x = resnets[i](x, timeEmbedding: timeEmbedding)

            if let attentions {
                x = attentions[i](
                    x, encoderX: encoderX, attentionMask: attentionMask,
                    encoderAttentionMask: encoderAttentionMask)
            }

            outputStates.append(x)
        }

        if let downsample {
            x = downsample(x)
            outputStates.append(x)
        }

        if let upsample {
            x = upsample(upsampleNearest(x))
            outputStates.append(x)
        }

        if let residualHiddenStates {
            return (x, outputStates, residualHiddenStates)
        } else {
            return (x, outputStates, [])
        }
    }
}

class UNetModel: Module {

    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    let timesteps: SinusoidalPositionalEncoding
    @ModuleInfo(key: "time_embedding") var timeEmbedding: TimestepEmbedding

    @ModuleInfo(key: "addition_embed_type") var addTimeProj: SinusoidalPositionalEncoding?
    @ModuleInfo(key: "add_embedding") var addEmbedding: TimestepEmbedding?

    @ModuleInfo(key: "down_blocks") var downBlocks: [UNetBlock2D]
    @ModuleInfo(key: "mid_blocks") var midBlocks: (ResnetBlock2D, Transformer2D, ResnetBlock2D)
    @ModuleInfo(key: "up_blocks") var upBlocks: [UNetBlock2D]

    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(configuration: UNetConfiguration) {
        let channels0 = configuration.blockOutChannels[0]

        self._convIn.wrappedValue = Conv2d(
            inputChannels: configuration.inputChannels, outputChannels: channels0,
            kernelSize: .init(configuration.convolutionInKernel),
            padding: .init((configuration.convolutionInKernel - 1) / 2))

        self.timesteps = SinusoidalPositionalEncoding(
            dimensions: channels0,
            minFrequency: exp(-log(10_000) + 2 * log(10_000) / Float(channels0)),
            maxFrequency: 1, scale: 1, cosineFirst: true, fullTurns: false)

        self._timeEmbedding.wrappedValue = TimestepEmbedding(
            inputChannels: channels0, timeEmbedDimensions: channels0 * 4)

        if configuration.additionEmbedType == "text_time",
            let additionTimeEmbedDimension = configuration.additionTimeEmbedDimension,
            let projectionClassEmbeddingsInputDimension = configuration
                .projectionClassEmbeddingsInputDimension
        {
            self._addTimeProj.wrappedValue = SinusoidalPositionalEncoding(
                dimensions: additionTimeEmbedDimension,
                minFrequency: exp(
                    -log(10_000) + 2 * log(10_000) / Float(additionTimeEmbedDimension)),
                maxFrequency: 1,
                scale: 1, cosineFirst: true, fullTurns: false)

            self._addEmbedding.wrappedValue = TimestepEmbedding(
                inputChannels: projectionClassEmbeddingsInputDimension,
                timeEmbedDimensions: channels0 * 4)
        }

        // make the downsampling blocks
        let downblockChannels = [channels0] + configuration.blockOutChannels
        self._downBlocks.wrappedValue = zip(downblockChannels, downblockChannels.dropFirst())
            .enumerated()
            .map { (i, pair) in
                let (inChannels, outChannels) = pair
                return UNetBlock2D(
                    inputChannels: inChannels,
                    outputChannels: outChannels,
                    timeEmbedChannels: channels0 * 4,
                    numLayers: configuration.layersPerBlock[i],
                    transformerLayersPerBlock: configuration.transformerLayersPerBlock[i],
                    numHeads: configuration.numHeads[i],
                    crossAttentionDimension: configuration.crossAttentionDimension[i],
                    resnetGroups: configuration.normNumGroups,
                    addDownSample: i < configuration.blockOutChannels.count - 1,
                    addUpSample: false,
                    addCrossAttention: configuration.downBlockTypes[i].contains("CrossAttn")
                )
            }

        // make the middle block
        let channelsLast = configuration.blockOutChannels.last!
        self._midBlocks.wrappedValue = (
            ResnetBlock2D(
                inputChannels: channelsLast,
                outputChannels: channelsLast,
                groupCount: configuration.normNumGroups,
                timeEmbedChannels: channels0 * 4
            ),
            Transformer2D(
                inputChannels: channelsLast,
                modelDimensions: channelsLast,
                encoderDimensions: configuration.crossAttentionDimension.last!,
                numHeads: configuration.numHeads.last!,
                numLayers: configuration.transformerLayersPerBlock.last!
            ),
            ResnetBlock2D(
                inputChannels: channelsLast,
                outputChannels: channelsLast,
                groupCount: configuration.normNumGroups,
                timeEmbedChannels: channels0 * 4
            )
        )

        // make the upsampling blocks
        let upblockChannels =
            [channels0] + configuration.blockOutChannels + [configuration.blockOutChannels.last!]
        self._upBlocks.wrappedValue =
            zip(upblockChannels, zip(upblockChannels.dropFirst(), upblockChannels.dropFirst(2)))
            .enumerated()
            .reversed()
            .map { (i, triple) in
                let (inChannels, (outChannels, prevOutChannels)) = triple
                return UNetBlock2D(
                    inputChannels: inChannels,
                    outputChannels: outChannels,
                    timeEmbedChannels: channels0 * 4,
                    previousOutChannels: prevOutChannels,
                    numLayers: configuration.layersPerBlock[i] + 1,
                    transformerLayersPerBlock: configuration.transformerLayersPerBlock[i],
                    numHeads: configuration.numHeads[i],
                    crossAttentionDimension: configuration.crossAttentionDimension[i],
                    resnetGroups: configuration.normNumGroups,
                    addDownSample: false,
                    addUpSample: i > 0,
                    addCrossAttention: configuration.upBlockTypes[i].contains("CrossAttn")
                )
            }

        self._convNormOut.wrappedValue = GroupNorm(
            groupCount: configuration.normNumGroups, dimensions: channels0, pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(
            inputChannels: channels0, outputChannels: configuration.outputChannels,
            kernelSize: .init(configuration.convolutionOutKernel),
            padding: .init((configuration.convolutionOutKernel - 1) / 2))
    }

    func callAsFunction(
        _ x: MLXArray, timestep: MLXArray, encoderX: MLXArray, attentionMask: MLXArray? = nil,
        encoderAttentionMask: MLXArray? = nil, textTime: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        // compute the time embeddings
        var temb = timesteps(timestep).asType(x.dtype)
        temb = timeEmbedding(temb)

        // add the extra textTime conditioning
        if let (textEmbedding, timeIds) = textTime,
            let addTimeProj, let addEmbedding
        {
            var emb = addTimeProj(timeIds).flattened(start: 1).asType(x.dtype)
            emb = concatenated([textEmbedding, emb], axis: -1)
            emb = addEmbedding(emb)
            temb = temb + emb
        }

        // preprocess the input
        var x = convIn(x)

        // run the downsampling part of the unet
        var residuals = [x]
        for block in self.downBlocks {
            let res: [MLXArray]
            (x, res, _) = block(
                x, encoderX: encoderX, timeEmbedding: temb, attentionMask: attentionMask,
                encoderAttentionMask: encoderAttentionMask)
            residuals.append(contentsOf: res)
        }

        // run the middle part of the unet
        x = midBlocks.0(x, timeEmbedding: temb)
        x = midBlocks.1(
            x, encoderX: encoderX, attentionMask: attentionMask,
            encoderAttentionMask: encoderAttentionMask)
        x = midBlocks.2(x, timeEmbedding: temb)

        // run the upsampling part of the unet
        for block in self.upBlocks {
            (x, _, residuals) = block(
                x, encoderX: encoderX, timeEmbedding: temb, attentionMask: attentionMask,
                encoderAttentionMask: encoderAttentionMask, residualHiddenStates: residuals)
        }

        // postprocess the output
        let dtype = x.dtype
        x = convNormOut(x)
        x = silu(x)
        x = convOut(x)

        return x
    }
}
