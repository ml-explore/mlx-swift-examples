import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum Qwen3VLLanguage {

    // MARK: - Rotary Embedding

    final class RotaryEmbedding {

        private let invFreq: MLXArray
        private let mropeSection: [Int]

        init(headDim: Int, base: Double, ropeScaling: Qwen3VLConfiguration.RoPEScaling?) {
            var freq = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
            freq = freq / Float(headDim)
            let baseArray = MLXArray(Float(base))
            self.invFreq = 1.0 / pow(baseArray, freq)
            self.mropeSection = ropeScaling?.mropeSection ?? [24, 20, 20]
           // print("[RotaryEmbedding] Initialized with headDim=\(headDim) base=\(base) mropeSection=\(self.mropeSection)")
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            // freqs shape: (3, bs, seq_len, head_dim // 2)
            // Extract the first dimension as the base output
            let freqs_t = freqs[0, 0..., 0..., 0...]  // (bs, seq_len, head_dim // 2)
            
            // Convert to mutable array by splitting along last dimension
            let dims = freqs_t.dim(-1)
            var slices: [MLXArray] = []
            
            for idx in 0..<dims {
                var slice = freqs_t[0..., 0..., idx]
                
                // Check if this index should be replaced from H or W dimensions
                // Python: for dim, offset in enumerate((1, 2), start=1)
                for (dimIndex, offset) in [(1, 1), (2, 2)] {
                    let end = min(mropeSection[dimIndex] * 3, dims)
                    if idx >= offset && idx < end && (idx - offset) % 3 == 0 {
                        // This index should come from freqs[dimIndex]
                        slice = freqs[dimIndex, 0..., 0..., idx]
                        break
                    }
                }
                
                slices.append(slice)
            }
            
            // Stack all slices back together along the last dimension
            return stacked(slices, axis: -1)
        }

        func callAsFunction(positionIds: MLXArray, dtype: MLX.DType) -> (MLXArray, MLXArray) {
            var positionIds = positionIds
            if positionIds.ndim == 2 {
                positionIds = positionIds[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds, repetitions: [3, 1, 1])
            }

            // Python uses matmul: inv_freq_expanded @ position_ids_expanded
            // But broadcasting achieves the same outer product:
            // (3, bs, seq_len, 1) * (1, 1, 1, inv_freq_len) → (3, bs, seq_len, inv_freq_len)
            let pos = positionIds.asType(.float32)
            var invFreq = self.invFreq.asType(.float32)
            invFreq = invFreq[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * invFreq
            freqs = applyInterleavedMRope(freqs)

            var emb = concatenated([freqs, freqs], axis: -1)
            let cosValues = cos(emb).asType(dtype)
            let sinValues = sin(emb).asType(dtype)
            return (cosValues, sinValues)
        }
    }

    static func applyMultimodalRotary(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        var cos = cos
        var sin = sin
        cos = expandedDimensions(cos, axis: 1)
        sin = expandedDimensions(sin, axis: 1)
        let qEmbedded = (q * cos) + (QwenVL.rotateHalf(q) * sin)
        let kEmbedded = (k * cos) + (QwenVL.rotateHalf(k) * sin)
        return (qEmbedded, kEmbedded)
    }

    // MARK: - Attention

    final class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            self.heads = config.numAttentionHeads
            self.kvHeads = config.numKeyValueHeads
            self.headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))

            rotaryEmbedding = RotaryEmbedding(
                headDim: headDim,
                base: config.ropeTheta,
                ropeScaling: config.ropeScaling)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(batch, length, heads, headDim)
            queries = qNorm(queries).transposed(0, 2, 1, 3)

            keys = keys.reshaped(batch, length, kvHeads, headDim)
            keys = kNorm(keys).transposed(0, 2, 1, 3)

            values = values.reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

            var kvSequenceLength = keys.dim(-2)
            var positionIds = positionIds

            if positionIds == nil {
                let offset = cache?.offset ?? 0
                // Python: kv_seq_len += cache.offset + 1
                // The +1 is because at position N, we can attend to 0..N (N+1 positions)
                kvSequenceLength += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + length, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                positionIds = base[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
            } else {
                // Python: kv_seq_len += cache.offset + 1 if cache is not None else 0
                if let cache {
                    kvSequenceLength += cache.offset + 1
                }
            }

            let (cosValues, sinValues) = rotaryEmbedding(positionIds: positionIds!, dtype: x.dtype)
            
            // Debug RoPE values during first few generation steps
//            if let cache, cache.offset >= 26 && cache.offset <= 28 {
//                print("[Attention] offset=\(cache.offset) positionIds.shape=\(positionIds!.shape) cos.shape=\(cosValues.shape) queries.shape=\(queries.shape)")
//            }
            
            (queries, keys) = Qwen3VLLanguage.applyMultimodalRotary(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                // Python: mask[..., :kv_seq_len]
                let slicedMask = mask[.ellipsis, 0 ..< kvSequenceLength]
                attentionMask = .array(slicedMask)
            } else {
                attentionMask = .none
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)

            let result = wo(output)
            
            // Debug attention output for first few generation steps
//            if let cache, cache.offset >= 26 && cache.offset <= 28 {
//                print("[Attention] offset=\(cache.offset) output mean=\(output.mean().item(Float.self)) result mean=\(result.mean().item(Float.self))")
//            }
//            
            return result
        }
    }

    // MARK: - Feed Forward

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    final class DecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            var residual = attention(inputLayerNorm(x), mask: mask, cache: cache, positionIds: positionIds)
            let hidden = x + residual
            residual = mlp(postAttentionLayerNorm(hidden))
            return hidden + residual
        }
    }

    final class Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            precondition(config.vocabSize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize)
            _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in DecoderLayer(config) }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?
        ) -> MLXArray {
            var hidden: MLXArray
            if let inputEmbeddings {
                hidden = inputEmbeddings
            } else if let inputIds {
                hidden = embedTokens(inputIds)
            } else {
                fatalError("Either input ids or embeddings must be provided")
            }

            var mask = mask
            if mask == nil {
                mask = createAttentionMask(h: hidden, cache: cache)
            }

            for (index, layer) in layers.enumerated() {
                let layerCache = cache?[index]
                hidden = layer(hidden, mask: mask, cache: layerCache, positionIds: positionIds)

                if let embeds = deepstackEmbeds, index < embeds.count,
                    let visualMask
                {
                    // Debug deepstack application during prefill
                    let cacheOffset = layerCache?.offset ?? 0
                    if cacheOffset == 0 || (cacheOffset > 150 && index == 0) {
                        let maskSum = visualMask.sum().item(Int.self)
                        print("[Model.callAsFunction] Layer \(index): Applying deepstack to \(maskSum) positions, hidden.shape=\(hidden.shape), embeds.shape=\(embeds[index].shape)")
                    }
                    
                    hidden = applyDeepstack(
                        hiddenStates: hidden,
                        visualMask: visualMask,
                        visualEmbeds: embeds[index])
                }
            }

            return norm(hidden)
        }

        private func applyDeepstack(
            hiddenStates: MLXArray,
            visualMask: MLXArray,
            visualEmbeds: MLXArray
        ) -> MLXArray {
            // visualMask is 1D [seq], convert to indices where mask is True
            let indices = maskIndices(visualMask)
            guard !indices.isEmpty else { return hiddenStates }
            
            let indexArray = MLXArray(indices.map { UInt32($0) })
            
            // Check stats before deepstack
            let beforeSlice = hiddenStates[0, indexArray[0], 0...]
            let beforeMean = beforeSlice.mean().item(Float.self)
            let beforeMin = beforeSlice.min().item(Float.self)
            let beforeMax = beforeSlice.max().item(Float.self)
            
            let embedMean = visualEmbeds.mean().item(Float.self)
            let embedMin = visualEmbeds.min().item(Float.self)
            let embedMax = visualEmbeds.max().item(Float.self)
            
            // Python: hidden_states[:, visual_indices, :] += visual_embeds
            // hidden_states is [batch, seq, hidden], visual_embeds is [num_visual, hidden]
            var result = hiddenStates
            result[0..., indexArray, 0...] = result[0..., indexArray, 0...] + visualEmbeds
            
            // Check stats after deepstack
            let afterSlice = result[0, indexArray[0], 0...]
            let afterMean = afterSlice.mean().item(Float.self)
            let afterMin = afterSlice.min().item(Float.self)
            let afterMax = afterSlice.max().item(Float.self)
            
            print("[applyDeepstack] Before: mean=\(beforeMean) range=[\(beforeMin), \(beforeMax)]")
            print("[applyDeepstack] Embeds: mean=\(embedMean) range=[\(embedMin), \(embedMax)]")
            print("[applyDeepstack] After:  mean=\(afterMean) range=[\(afterMin), \(afterMax)]")

            return result
        }

        private func maskIndices(_ mask: MLXArray) -> [Int] {
            // For 1D boolean mask, return indices where True
            let bools = mask.asType(.bool).asArray(Bool.self)
            var indices: [Int] = []
            indices.reserveCapacity(bools.count)
            for (idx, value) in bools.enumerated() where value {
                indices.append(idx)
            }
            return indices
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {

        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: Qwen3VLConfiguration
        let textConfig: Qwen3VLConfiguration.TextConfiguration
        var kvHeads: [Int]

        private var ropeDeltas: MLXArray? = nil

        init(_ config: Qwen3VLConfiguration) {
            self.config = config
            self.textConfig = config.textConfiguration
            self.model = Model(config.textConfiguration)
            self.kvHeads = Array(repeating: config.textConfiguration.numKeyValueHeads,
                                 count: config.textConfiguration.numHiddenLayers)

            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabSize,
                    bias: false)
            }
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds providedPositionIds: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?,
            pixelValues: MLXArray?,
            imageGridTHW: [THW]?,
            videoGridTHW: [THW]?
        ) -> LMOutput {
            if pixelValues != nil {
                ropeDeltas = nil
            }

            var positionIds = providedPositionIds
            
            if positionIds == nil && (mask == nil || mask?.ndim == 2) {
                // Python: recalculate if offset==0 OR ropeDeltas is None OR cache is None
                if (cache?.first?.offset ?? 0) == 0 || ropeDeltas == nil || cache == nil {
                    if let inputIds {
                        let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
                            inputIds: inputIds,
                            imageGridTHW: imageGridTHW,
                            videoGridTHW: videoGridTHW,
                            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                            imageTokenId: config.imageTokenIndex,
                            videoTokenId: config.videoTokenIndex,
                            visionStartTokenId: config.visionStartTokenId,
                            attentionMask: mask)
                        
                        positionIds = computed
                        ropeDeltas = deltas
                    } else if let cache, ropeDeltas == nil {
                        // Text-only generation: no inputIds, just use sequential positions
                        let batch = inputEmbeddings!.dim(0)
                        let seqLength = inputEmbeddings!.dim(1)
                        let currentOffset = cache.first?.offset ?? 0
                        
                        var base = MLXArray(0..<seqLength).asType(.int32)
                        base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                        let offsetValue = MLXArray(currentOffset).asType(.int32)
                        base = base + offsetValue
                        
                        // Expand to 3D for MRoPE: [batch, seq] -> [3, batch, seq]
                        positionIds = base[.newAxis, 0..., 0...]
                        positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
                    }
                } else if let cache, let ropeDeltas {
                    // Python: delta = cache[-1].offset + self.rope_deltas
                    // Python: position_ids = mx.arange(seq_length) + delta
                    // Python: position_ids = mx.broadcast_to(position_ids, (3, batch, seq))
                    let batch = (inputIds ?? inputEmbeddings!).dim(0)
                    let seqLength = (inputIds ?? inputEmbeddings!).dim(1)
                    
                    // CRITICAL: Python uses cache[-1].offset (LAST element), not cache[0].offset!
                    let lastCacheOffset = cache.last?.offset ?? 0
                    
                    // Python: delta = cache[-1].offset + self.rope_deltas
                    var delta = MLXArray(lastCacheOffset).asType(.int32) + ropeDeltas.asType(.int32)
                    
                    // Python: position_ids = mx.arange(seq_length).reshape(1, seq_length)
                    var base = MLXArray(0..<seqLength).asType(.int32)
                    base = base[.newAxis, 0...]  // Shape: [1, seq_length]
                    base = broadcast(base, to: [batch, seqLength])
                    
                    // Broadcast delta across the batch axis (Python repeats along axis 0)
                    if delta.dim(0) == 1 && batch > 1 {
                        delta = repeated(delta, count: batch, axis: 0)
                    }
                    
                    // Python: position_ids = mx.add(position_ids, delta)
                    base = base + delta
                    
                    // Expand to 3D: [batch, seq] -> [3, batch, seq]
                    positionIds = base[.newAxis, 0..., 0...]
                    positionIds = broadcast(positionIds!, to: [3, batch, seqLength])                    
                }
            }
            
            // CRITICAL: Python does NOT pass mask to model()!
            // The Model creates its own causal mask. Passing a 2D mask prevents this.
            var output = model(
                inputIds,
                cache: cache,
                inputEmbeddings: inputEmbeddings,
                mask: nil,  // Let Model create causal mask
                positionIds: positionIds,
                visualMask: visualMask,
                deepstackEmbeds: deepstackEmbeds)

            if let lmHead {
                output = lmHead(output)
            } else {
                output = model.embedTokens.asLinear(output)
            }

            return LMOutput(logits: output)
        }

        private func getRoPEIndex(
            inputIds: MLXArray,
            imageGridTHW: [THW]?,
            videoGridTHW: [THW]?,
            attentionMask: MLXArray?
        ) -> (MLXArray, MLXArray) {
            let batchSize = inputIds.dim(0)
            let seqLength = inputIds.dim(1)
            let spatialMerge = config.visionConfiguration.spatialMergeSize

            var positionStorage = Array(
                repeating: Array(
                    repeating: Array(repeating: Float(0), count: seqLength), count: batchSize),
                count: 3)
            var deltas = Array(repeating: Float(0), count: batchSize)

            let attentionDefaults: [[Int32]] = {
                if let attentionMask {
                    var rows: [[Int32]] = []
                    rows.reserveCapacity(batchSize)
                    for b in 0..<batchSize {
                        rows.append(attentionMask[b, 0...].asArray(Int32.self))
                    }
                    return rows
                } else {
                    let onesRow = Array(repeating: Int32(1), count: seqLength)
                    return Array(repeating: onesRow, count: batchSize)
                }
            }()

            var imageIndex = 0
            var videoIndex = 0

            for batch in 0..<batchSize {
                var tokens = inputIds[batch, 0...].asArray(Int.self)
                let maskRow = attentionDefaults[batch]

                // Only iterate up to maskRow length to avoid index out of bounds
                for idx in maskRow.indices where maskRow[idx] == 0 {
                    tokens[idx] = 0
                }

                // Count the number of actual images/videos from the grids, not token occurrences
                // Each image may have multiple imageTokenId tokens (one per patch)
                let imageCount = (imageGridTHW?.count ?? 0) - imageIndex
                let videoCount = (videoGridTHW?.count ?? 0) - videoIndex

                var segments: [[[Int]]] = []
                var st = 0
                var remainingImages = imageCount
                var remainingVideos = videoCount
                var sequentialTokenPosition = 0  // Track actual token count for text positioning

                func currentMax() -> Int {
                    guard let last = segments.last else { return -1 }
                    return last.flatMap { $0 }.max() ?? -1
                }

                func appendText(length: Int) {
                    guard length > 0 else { return }
                    let startIndex = sequentialTokenPosition
                    var segment = Array(repeating: Array(repeating: 0, count: length), count: 3)
                    for axis in 0..<3 {
                        for idx in 0..<length {
                            segment[axis][idx] = startIndex + idx
                        }
                    }
                    segments.append(segment)
                    sequentialTokenPosition += length
                }

                func appendVisual(grid: THW, textLength: Int, startIndex: Int) {
                    let t = grid.t
                    let h = grid.h / spatialMerge
                    let w = grid.w / spatialMerge
                    let patchCount = max(t * h * w, 0)
                    guard patchCount > 0 else { return }

                    var segment = Array(repeating: Array(repeating: 0, count: patchCount), count: 3)
                    var cursor = 0
                    for ti in 0..<t {
                        for hi in 0..<h {
                            for wi in 0..<w {
                                segment[0][cursor] = ti
                                segment[1][cursor] = hi
                                segment[2][cursor] = wi
                                cursor += 1
                            }
                        }
                    }

                    let offset = sequentialTokenPosition
                    for axis in 0..<3 {
                        for idx in 0..<patchCount {
                            segment[axis][idx] += offset
                        }
                    }
                    segments.append(segment)
                    sequentialTokenPosition += patchCount
                }

                for _ in 0..<(imageCount + videoCount) {
                    if st >= tokens.count { break }

                    let imageSearch = remainingImages > 0
                        ? tokens[st...].firstIndex(of: config.imageTokenId)
                        : nil
                    let videoSearch = remainingVideos > 0
                        ? tokens[st...].firstIndex(of: config.videoTokenId)
                        : nil

                    if imageSearch == nil && videoSearch == nil {
                        break
                    }

                    let useImage: Bool
                    let endIndex: Int
                    if let imageIndexValue = imageSearch, let videoIndexValue = videoSearch {
                        if imageIndexValue <= videoIndexValue {
                            useImage = true
                            endIndex = imageIndexValue
                        } else {
                            useImage = false
                            endIndex = videoIndexValue
                        }
                    } else if let imageIndexValue = imageSearch {
                        useImage = true
                        endIndex = imageIndexValue
                    } else if let videoIndexValue = videoSearch {
                        useImage = false
                        endIndex = videoIndexValue
                    } else {
                        break
                    }

                    let textLength = max(endIndex - st, 0)
                    appendText(length: textLength)

                    let startIndex = currentMax() + 1 - textLength

                    if useImage {
                        guard let grids = imageGridTHW, imageIndex < grids.count else { break }
                        appendVisual(grid: grids[imageIndex], textLength: textLength, startIndex: startIndex)
                        imageIndex += 1
                        remainingImages -= 1
                    } else {
                        guard let grids = videoGridTHW, videoIndex < grids.count else { break }
                        appendVisual(grid: grids[videoIndex], textLength: textLength, startIndex: startIndex)
                        videoIndex += 1
                        remainingVideos -= 1
                    }

                    let grid = useImage ? imageGridTHW?[imageIndex - 1] : videoGridTHW?[videoIndex - 1]
                    let patchCount: Int
                    if let grid {
                        patchCount = (grid.t) * (grid.h / spatialMerge) * (grid.w / spatialMerge)
                    } else {
                        patchCount = 0
                    }

                    st = min(endIndex + patchCount, tokens.count)
                }

                if st < tokens.count {
                    appendText(length: tokens.count - st)
                }

                var combined = Array(repeating: [Int](), count: 3)
                for segment in segments {
                    for axis in 0..<3 {
                        combined[axis].append(contentsOf: segment[axis])
                    }
                }

                for axis in 0..<3 {
                    if combined[axis].count < seqLength {
                        let startIndex = (combined[axis].last ?? -1) + 1
                        let needed = seqLength - combined[axis].count
                        combined[axis].append(contentsOf: (0..<needed).map { startIndex + $0 })
                    } else if combined[axis].count > seqLength {
                        combined[axis] = Array(combined[axis].prefix(seqLength))
                    }
                }

                for idx in 0..<seqLength {
                    if maskRow[idx] == 0 {
                        for axis in 0..<3 {
                            positionStorage[axis][batch][idx] = Float(idx)
                        }
                    } else {
                        for axis in 0..<3 {
                            positionStorage[axis][batch][idx] = Float(combined[axis][idx])
                        }
                    }
                }

                let maxPosition = combined[0].max() ?? (seqLength - 1)
                let deltaValue = max(maxPosition + 1 - seqLength, 0)
                deltas[batch] = Float(deltaValue)
                
                if batch == 0 && seqLength > 200 {
                    print("[getRoPEIndex] LARGE sequence: seqLength=\(seqLength) maxPos=\(maxPosition) delta=\(deltaValue)")
                    print("[getRoPEIndex] sequentialTokenPosition ended at: \(sequentialTokenPosition)")
                }
            }

            let flatPositions = positionStorage.flatMap { $0.flatMap { $0 } }
            let positionArray = MLXArray(flatPositions, [3, batchSize, seqLength]).asType(.int32)
            let deltaArray = MLXArray(deltas, [batchSize, 1]).asType(.int32)
            return (positionArray, deltaArray)
        }
    }
}
