// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum Qwen3VLVision {

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let first = x[.ellipsis, 0 ..< half]
        let second = x[.ellipsis, half...]
        return concatenated([-second, first], axis: -1)
    }

    static func applyRotary(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
        var cosVals = cos(freqs)
        var sinVals = sin(freqs)

        cosVals = expandedDimensions(cosVals, axis: 1)
        cosVals = tiled(cosVals, repetitions: [1, 1, 2])
        cosVals = expandedDimensions(cosVals, axis: 0)

        sinVals = expandedDimensions(sinVals, axis: 1)
        sinVals = tiled(sinVals, repetitions: [1, 1, 2])
        sinVals = expandedDimensions(sinVals, axis: 0)

        let rotated = (tensor * cosVals) + (rotateHalf(tensor) * sinVals)
        return rotated.asType(tensor.dtype)
    }

    final class VisionRotaryEmbedding {
        let dimension: Int
        let theta: Float

        init(dimension: Int, theta: Float = 10_000) {
            self.dimension = dimension
            self.theta = theta
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let invFreq = 1.0 / pow(
                MLXArray(theta),
                MLXArray(stride(from: 0, to: dimension, by: 2)).asType(.float32) / Float(dimension)
            )
            let seq = MLXArray(0..<sequenceLength).asType(invFreq.dtype)
            return outer(seq, invFreq)
        }
    }

    final class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo(key: "proj") var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let hiddenSize: Int

        init(
            patchSize: Int,
            temporalPatchSize: Int,
            inChannels: Int,
            hiddenSize: Int
        ) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.hiddenSize = hiddenSize

            let kernel = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            _proj.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: hiddenSize,
                kernelSize: kernel,
                stride: kernel,
                bias: true
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var states = x.reshaped(
                -1,
                inChannels,
                temporalPatchSize,
                patchSize,
                patchSize
            ).movedAxis(source: 1, destination: 4)

            states = proj(states)
            states = states.reshaped(-1, hiddenSize)
            return states
        }
    }

    final class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        let usePostShuffleNorm: Bool

        @ModuleInfo(key: "norm") var norm: LayerNorm
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear
        @ModuleInfo(key: "act") var activation: GELU

        init(config: Qwen3VLConfiguration.VisionConfiguration, usePostShuffleNorm: Bool) {
            self.hiddenSize = config.hiddenSize * (config.spatialMergeSize * config.spatialMergeSize)
            self.usePostShuffleNorm = usePostShuffleNorm

            let normDim = usePostShuffleNorm ? hiddenSize : config.hiddenSize
            _norm.wrappedValue = LayerNorm(dimensions: normDim, eps: 1e-6)
            _linear1.wrappedValue = Linear(hiddenSize, hiddenSize)
            _linear2.wrappedValue = Linear(hiddenSize, config.outHiddenSize)
            _activation.wrappedValue = GELU()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var states = x
            if usePostShuffleNorm {
                states = states.reshaped(-1, hiddenSize)
            }
            states = norm(states)
            states = states.reshaped(-1, hiddenSize)
            states = linear1(states)
            states = activation(states)
            states = linear2(states)
            return states
        }
    }

    final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        init(dim: Int, numHeads: Int) {
            self.numHeads = numHeads
            self.headDim = dim / numHeads
            self.scale = pow(Float(headDim), -0.5)

            _qkv.wrappedValue = Linear(dim, 3 * dim, bias: true)
            _proj.wrappedValue = Linear(dim, dim)
        }

        func callAsFunction(
            _ x: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)

            var qkvStates = qkv(x)
            qkvStates = qkvStates.reshaped(sequenceLength, 3, numHeads, headDim)
            qkvStates = qkvStates.transposed(1, 0, 2, 3)

            let parts = split(qkvStates, parts: 3, axis: 0)
            var queries = parts[0][0, 0..., 0..., 0...]
            var keys = parts[1][0, 0..., 0..., 0...]
            var values = parts[2][0, 0..., 0..., 0...]

            queries = applyRotary(queries, freqs: rotaryPosEmb)
            keys = applyRotary(keys, freqs: rotaryPosEmb)

            queries = queries.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)

            var mask = ones([1, sequenceLength, sequenceLength], dtype: queries.dtype)
            mask = mask * MLXArray(-1e9, dtype: queries.dtype)

            let seqlens = cuSeqlens.asArray(Int.self)
            for idx in 1..<seqlens.count {
                let start = seqlens[idx - 1]
                let end = seqlens[idx]
                mask[0..., start ..< end, start ..< end] = MLXArray(0, dtype: queries.dtype)
            }

            let attended = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .array(mask)
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)

            return proj(attended)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear
        @ModuleInfo(key: "act") var activation: GELU

        init(dim: Int, hiddenDim: Int) {
            _linear1.wrappedValue = Linear(dim, hiddenDim, bias: true)
            _linear2.wrappedValue = Linear(hiddenDim, dim, bias: true)
            _activation.wrappedValue = GELU(approximation: .fast)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(activation(linear1(x)))
        }
    }

    final class VisionBlock: Module {
        @ModuleInfo(key: "norm1") var norm1: LayerNorm
        @ModuleInfo(key: "norm2") var norm2: LayerNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        init(_ config: Qwen3VLConfiguration.VisionConfiguration) {
            _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _attention.wrappedValue = Attention(dim: config.hiddenSize, numHeads: config.numHeads)
            _mlp.wrappedValue = MLP(dim: config.hiddenSize, hiddenDim: config.intermediateSize)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray
        ) -> MLXArray {
            var states = hiddenStates
            states = states + attention(norm1(states), cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryPosEmb)
            states = states + mlp(norm2(states))
            return states
        }
    }

    final class VisionModel: Module {

        let config: Qwen3VLConfiguration.VisionConfiguration
        let spatialMergeSize: Int
        let numGridPerSide: Int

        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ModuleInfo(key: "rotary_pos_emb") var rotaryEmbedding: VisionRotaryEmbedding
        @ModuleInfo(key: "pos_embed") var posEmbed: Embedding
        @ModuleInfo(key: "blocks") var blocks: [VisionBlock]
        @ModuleInfo(key: "merger") var merger: PatchMerger
        @ModuleInfo(key: "deepstack_merger_list") var deepstackMergers: [PatchMerger]
        let deepstackVisualIndexes: [Int]

        init(_ config: Qwen3VLConfiguration.VisionConfiguration) {
            self.config = config
            self.spatialMergeSize = config.spatialMergeSize
            self.numGridPerSide = Int(sqrt(Double(config.numPositionEmbeddings)))
            self.deepstackVisualIndexes = config.deepstackVisualIndexes

            _patchEmbed.wrappedValue = PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                hiddenSize: config.hiddenSize)

            let headDim = config.hiddenSize / config.numHeads
            _rotaryEmbedding.wrappedValue = VisionRotaryEmbedding(dimension: headDim / 2)

            _posEmbed.wrappedValue = Embedding(
                embeddingCount: config.numPositionEmbeddings,
                dimensions: config.hiddenSize)

            _blocks.wrappedValue = (0..<config.depth).map { _ in VisionBlock(config) }
            _merger.wrappedValue = PatchMerger(config: config, usePostShuffleNorm: false)

            _deepstackMergers.wrappedValue = config.deepstackVisualIndexes.map { _ in
                PatchMerger(config: config, usePostShuffleNorm: true)
            }
        }

        private func rotaryPositionEmbedding(_ grids: [THW]) -> MLXArray {
            guard let maxHW = grids.map({ max($0.h, $0.w) }).max(), maxHW > 0 else {
                return MLXArray.zeros([1, 1], dtype: .float32)
            }

            let freqTable = rotaryEmbedding(sequenceLength: maxHW)
            let halfDim = freqTable.dim(-1)
            
            let merge = spatialMergeSize
            var positionPairs: [[Int32]] = [[], []]  // [h_positions, w_positions]
            
            for grid in grids {
                let mergedH = grid.h / merge
                let mergedW = grid.w / merge
                
                // Create block and intra-block indices
                let blockRows = Array(0..<mergedH)
                let blockCols = Array(0..<mergedW)
                let intraIndices = Array(0..<merge)
                
                // For each temporal frame
                for _ in 0..<grid.t {
                    // Generate all position pairs for this frame
                    for mh in blockRows {
                        for mw in blockCols {
                            for ih in intraIndices {
                                for iw in intraIndices {
                                    let hPos = Int32(mh * merge + ih)
                                    let wPos = Int32(mw * merge + iw)
                                    positionPairs[0].append(hPos)
                                    positionPairs[1].append(wPos)
                                }
                            }
                        }
                    }
                }
            }
            
            guard !positionPairs[0].isEmpty else {
                return MLXArray.zeros([0, halfDim * 2], dtype: freqTable.dtype)
            }
            
            // Convert to MLX arrays
            let hArray = MLXArray(positionPairs[0])
            let wArray = MLXArray(positionPairs[1])
            
            // Lookup embeddings using array indexing
            let hEmbeds = freqTable[hArray, 0...]
            let wEmbeds = freqTable[wArray, 0...]
            
            // Concatenate height and width embeddings
            return concatenated([hEmbeds, wEmbeds], axis: -1)
        }

        private func positionalEmbeddings(_ grids: [THW]) -> MLXArray {
            let hiddenSize = config.hiddenSize
            let maxIndex = numGridPerSide - 1
            
            // Step 1: Collect all indices and weights from all grids using MLX ops
            var cornerIndices: [[MLXArray]] = Array(repeating: [], count: 4)
            var cornerWeights: [[MLXArray]] = Array(repeating: [], count: 4)
            var gridSizes: [Int] = []
            
            for grid in grids {
                let h = grid.h
                let w = grid.w
                gridSizes.append(h * w)
                
                // Create linspace indices using broadcasting
                var hLinspace = MLXArray(0..<h).asType(.float32)
                hLinspace = hLinspace * MLXArray(Float(maxIndex)) / MLXArray(Float(max(1, h - 1)))
                
                var wLinspace = MLXArray(0..<w).asType(.float32)
                wLinspace = wLinspace * MLXArray(Float(maxIndex)) / MLXArray(Float(max(1, w - 1)))
                
                // Get floor/ceil and deltas
                let hFloor = hLinspace.asType(.int32)
                let hCeil = minimum(hFloor + 1, maxIndex)
                let dh = hLinspace - hFloor.asType(.float32)
                
                let wFloor = wLinspace.asType(.int32)
                let wCeil = minimum(wFloor + 1, maxIndex)
                let dw = wLinspace - wFloor.asType(.float32)
                
                // Broadcast to create meshgrid
                let hFloorExpanded = expandedDimensions(hFloor, axis: 1)  // (h, 1)
                let hCeilExpanded = expandedDimensions(hCeil, axis: 1)
                let wFloorExpanded = expandedDimensions(wFloor, axis: 0)  // (1, w)
                let wCeilExpanded = expandedDimensions(wCeil, axis: 0)
                
                let baseH = hFloorExpanded * numGridPerSide
                let baseHCeil = hCeilExpanded * numGridPerSide
                
                // Compute 4 corner indices
                cornerIndices[0].append((baseH + wFloorExpanded).flattened())
                cornerIndices[1].append((baseH + wCeilExpanded).flattened())
                cornerIndices[2].append((baseHCeil + wFloorExpanded).flattened())
                cornerIndices[3].append((baseHCeil + wCeilExpanded).flattened())
                
                // Compute bilinear weights
                let dhExpanded = expandedDimensions(dh, axis: 1)
                let dwExpanded = expandedDimensions(dw, axis: 0)
                
                cornerWeights[0].append(((1 - dhExpanded) * (1 - dwExpanded)).flattened())
                cornerWeights[1].append(((1 - dhExpanded) * dwExpanded).flattened())
                cornerWeights[2].append((dhExpanded * (1 - dwExpanded)).flattened())
                cornerWeights[3].append((dhExpanded * dwExpanded).flattened())
            }
            
            guard !cornerIndices[0].isEmpty else {
                return MLXArray.zeros([0, hiddenSize], dtype: posEmbed.weight.dtype)
            }
            
            // Step 2: Batch embedding lookup
            let indicesTensors = cornerIndices.map { concatenated($0, axis: 0).asType(.int32) }
            let weightsTensors = cornerWeights.map { concatenated($0, axis: 0).asType(posEmbed.weight.dtype) }
            
            let totalPatches = indicesTensors[0].dim(0)
            var patchPosEmbeds = MLXArray.zeros([totalPatches, hiddenSize], dtype: posEmbed.weight.dtype)
            
            for cornerIdx in 0..<4 {
                let cornerEmbeds = posEmbed(indicesTensors[cornerIdx])
                let weighted = cornerEmbeds * expandedDimensions(weightsTensors[cornerIdx], axis: -1)
                patchPosEmbeds = patchPosEmbeds + weighted
            }
            
            // Step 3: Split by grid (like Python lines 344-349)
            var patchPosEmbedsSplit: [MLXArray] = []
            var offset = 0
            
            for size in gridSizes {
                let slice = patchPosEmbeds[offset..<(offset + size), 0...]
                patchPosEmbedsSplit.append(slice)
                offset += size
            }
            
            // Step 4: Process each grid (like Python lines 354-371)
            var resultEmbeds: [MLXArray] = []
            let merge = spatialMergeSize
            
            for (gridIdx, grid) in grids.enumerated() {
                let posEmbed = patchPosEmbedsSplit[gridIdx]
                let h = grid.h
                let w = grid.w
                let t = grid.t
                
                let featureDim = posEmbed.dim(-1)
                
                // Repeat for temporal dimension
                var temporalEmbeds = tiled(posEmbed, repetitions: [t, 1])
                
                // Reshape for merge pattern
                temporalEmbeds = temporalEmbeds.reshaped(t, h, w, featureDim)
                temporalEmbeds = temporalEmbeds.reshaped(
                    t,
                    h / merge,
                    merge,
                    w / merge,
                    merge,
                    featureDim
                )
                temporalEmbeds = temporalEmbeds.transposed(0, 1, 3, 2, 4, 5)
                temporalEmbeds = temporalEmbeds.reshaped(-1, featureDim)
                
                resultEmbeds.append(temporalEmbeds)
            }
            
            return concatenated(resultEmbeds, axis: 0)
        }

        private func cumulativeSequenceLengths(_ grids: [THW]) -> MLXArray {
            var prefix: [Int] = [0]
            for grid in grids {
                let perFrame = grid.h * grid.w  // Full patch count per frame
                for _ in 0..<grid.t {
                    prefix.append((prefix.last ?? 0) + perFrame)
                }
            }
            return MLXArray(prefix)
        }

        func callAsFunction(_ pixelValues: MLXArray, gridTHW: [THW]) -> (MLXArray, [MLXArray]) {
            var hiddenStates = patchEmbed(pixelValues)
            
            let posEmbeds = positionalEmbeddings(gridTHW)
            hiddenStates = hiddenStates + posEmbeds

            let rotaryEmbeds = rotaryPositionEmbedding(gridTHW)
            let cuSeqlens = cumulativeSequenceLengths(gridTHW)

            var deepstackOutputs: [MLXArray] = []

            for (index, block) in blocks.enumerated() {
                hiddenStates = block(hiddenStates, cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryEmbeds)
                if let dsIndex = deepstackVisualIndexes.firstIndex(of: index) {
                    let feature = deepstackMergers[dsIndex](hiddenStates)
                    deepstackOutputs.append(feature)
                }
            }

            hiddenStates = merger(hiddenStates)
            return (hiddenStates, deepstackOutputs)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                } else if key.contains("patch_embed.proj.weight") {
                    if value.ndim == 5 && value.dim(-1) == config.inChannels {
                        sanitized[key] = value
                    } else {
                        sanitized[key] = value.transposed(0, 2, 3, 4, 1)
                    }
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }
    }
}
