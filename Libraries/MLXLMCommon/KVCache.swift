// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Implementation of KV cache functionality for MLX Swift
///
///
/// ## Quantized Cache Usage
///
/// **Standard caches:**
/// ```swift
/// let cache = KVCacheSimple()
/// let (keys, values) = cache.update(keys: keys, values: values)
/// let output = MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: values, ...)
/// ```
///
/// **Quantized cache:**
/// ```swift
/// let quantizedCache = QuantizedKVCache(groupSize: 64, bits: 4)
/// let (qKeys, qValues) = quantizedCache.updateAndFetchQuantized(keys: keys, values: values)
///
/// let output = quantizedScaledDotProductAttention(
///     queries: queries,
///     quantizedKeys: qKeys,
///     quantizedValues: qValues,
///     scale: scale,
///     mask: mask,
///     groupSize: quantizedCache.groupSize,
///     bits: quantizedCache.bits
/// )
/// ```
///
/// Interface for Key/Value cache for LLMs.
///
/// See ``LanguageModel/newCache(parameters:)``
public protocol KVCache: Evaluatable {
    /// get the current offset
    var offset: Int { get }

    /// get the maximum size (if any)
    var maxSize: Int? { get }

    /// update the cache with new keys and values and return all keys/values
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// get the current state for serialization
    var state: [MLXArray] { get set }

    /// get/set metadata state as string array for serialization
    var metaState: [String] { get set }

    /// whether this cache can be trimmed
    var isTrimmable: Bool { get }

    /// trim n tokens from the cache, returning actual number trimmed
    @discardableResult
    func trim(_ n: Int) -> Int
}

/// Protocol for caches that support efficient quantized operations
///
/// **Usage Example:**
/// ```swift
/// // Efficient quantized path
/// if let quantizedCache = cache as? QuantizedKVCacheProtocol {
///     let (qKeys, qValues) = quantizedCache.updateAndFetchQuantized(keys: k, values: v)
///     // Use native quantized operations
///     let scores = quantizedMatmul(queries, w: qKeys.0, scales: qKeys.1, biases: qKeys.2, ...)
/// } else {
///     // Regular path
///     let (k, v) = cache.update(keys: k, values: v)
///     let output = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, ...)
/// }
/// ```
public protocol QuantizedKVCacheProtocol: KVCache {
    /// The quantization group size used
    var groupSize: Int { get }

    /// The number of quantization bits used
    var bits: Int { get }

    /// Update cache and return quantized tuples for maximum efficiency
    ///
    /// - Parameters:
    ///   - keys: New key data to add to cache
    ///   - values: New value data to add to cache
    /// - Returns: Quantized tuples (keys, values) as ((weight, scales, biases), (weight, scales, biases))
    func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray), (MLXArray, MLXArray, MLXArray)
    )

    /// Get current quantized state without updating
    ///
    /// Useful for accessing cached data without adding new tokens.
    /// - Returns: Current quantized state, or nil if cache is empty
    func getQuantizedState() -> ((MLXArray, MLXArray, MLXArray), (MLXArray, MLXArray, MLXArray))?
}

/// Base cache implementation providing default behaviors
open class BaseKVCache: KVCache {
    public var offset: Int = 0
    public var maxSize: Int? { nil }

    public func innerState() -> [MLXArray] { [] }

    open func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("update(keys:values:) must be implemented by subclass")
    }

    open var state: [MLXArray] {
        get { [] }
        set {
            if !newValue.isEmpty {
                fatalError("This cache has no state but a state was set.")
            }
        }
    }

    open var metaState: [String] {
        get {
            // Python base class returns empty string, but we return empty array for Swift compatibility
            // This is handled in the save/load functions
            []
        }
        set {
            if !newValue.isEmpty {
                fatalError("This cache has no meta_state but a meta_state was set.")
            }
        }
    }

    open var isTrimmable: Bool { false }

    @discardableResult
    open func trim(_ n: Int) -> Int { 0 }
}

func createCausalMask(n: Int, offset: Int, windowSize: Int? = nil) -> MLXArray {
    var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
    linds = linds[0..., .newAxis]
    rinds = rinds[.newAxis]
    var mask = linds .>= rinds

    if let windowSize {
        mask = mask & (linds .< rinds + windowSize)
    }

    return mask
}

/// Create an attention mask using the parameters from the KVCache.
///
/// See also ``MultiHeadAttention/createAdditiveCausalMask(_:dtype:)`` -- same idea
/// but doesn't honor the cache offset.
@_disfavoredOverload
public func createAttentionMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
    let t = h.dim(1)
    if t > 1 {
        var offset = 0
        if let c = cache?.first {
            offset = c.offset
        }
        return createCausalMask(n: t, offset: offset)
    }
    return nil
}

public func createAttentionMask(h: MLXArray, cache: [KVCache]?, returnArray: Bool = false)
    -> MLXFast.ScaledDotProductAttentionMaskMode
{
    let t = h.dim(1)
    if t > 1 {
        var returnArray = returnArray
        var offset = 0
        var windowSize: Int? = nil
        if let c = cache?.first {
            offset = c.offset
            if let maxSize = c.maxSize {
                windowSize = maxSize
                offset = min(maxSize, offset)
                if !returnArray {
                    returnArray = offset + t > maxSize
                }
            }
        }

        if returnArray {
            return .array(createCausalMask(n: t, offset: offset, windowSize: windowSize))
        } else {
            return .causal
        }
    }
    return .none
}

/// Standard KV cache implementation based on Python's KVCache
/// See https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/base.py#L11
public class KVCacheSimple: BaseKVCache, CustomDebugStringConvertible {
    internal var keys: MLXArray?
    internal var values: MLXArray?
    public var step = 256

    public override init() {
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        let reset =
            if let currentKeys = self.keys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.keys == nil
            }
        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        self.offset += keys.dim(2)

        self.keys?[.ellipsis, previous ..< self.offset, 0...] = keys
        self.values?[.ellipsis, previous ..< self.offset, 0...] = values

        let returnedKeys = self.keys![.ellipsis, ..<self.offset, 0...]
        let returnedValues = self.values![.ellipsis, ..<self.offset, 0...]

        return (returnedKeys, returnedValues)
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if offset == keys.dim(2) {
                return [keys, values]
            } else {
                return [
                    keys[.ellipsis, ..<offset, 0...],
                    values[.ellipsis, ..<offset, 0...],
                ]
            }
        }
        set {
            guard newValue.count == 2 else {
                fatalError("KVCacheSimple state must have exactly 2 arrays (keys, values)")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            self.offset = self.keys!.dim(2)
        }
    }

    public override var metaState: [String] {
        get { [] }
        set {
            if !newValue.isEmpty {
                fatalError("KVCacheSimple should not have metaState.")
            }
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    /// Convert to quantized cache for maximum efficiency
    ///
    /// Use `updateAndFetchQuantized()` and `quantizedScaledDotProductAttention()` for zero-overhead operation.
    public func toQuantized(groupSize: Int = 64, bits: Int = 4) -> QuantizedKVCache {
        let quantizedCache = QuantizedKVCache(groupSize: groupSize, bits: bits)
        quantizedCache.offset = self.offset

        if let keys = self.keys, let values = self.values {
            // Quantize the current keys and values
            let currentKeys = keys[.ellipsis, ..<offset, 0...]
            let currentValues = values[.ellipsis, ..<offset, 0...]

            let quantizedKeys = quantized(currentKeys, groupSize: groupSize, bits: bits)
            let quantizedValues = quantized(currentValues, groupSize: groupSize, bits: bits)

            // Set the quantized state
            quantizedCache.state = [
                quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases,
                quantizedValues.wq, quantizedValues.scales, quantizedValues.biases,
            ]
        }

        return quantizedCache
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) \(Unmanaged.passUnretained(self).toOpaque()), offset: \(offset), step: \(step), keys: \(keys?.shape.description ?? "-"), values: \(values?.shape.description ?? "-")"
    }
}

/// Rotating KV cache for sliding window attention
public class RotatingKVCache: BaseKVCache, CustomDebugStringConvertible {
    private var keep: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    // TODO: `offset` from the Python implementation is not implemented here. Do we need it?
    private var maxCacheSize: Int?
    private var step: Int
    private var idx: Int = 0

    public override var maxSize: Int? { maxCacheSize }

    public init(maxSize: Int?, keep: Int = 0, step: Int = 256) {
        self.maxCacheSize = maxSize
        self.keep = keep
        self.step = step
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    private func trim(trimSize: Int, _ array: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var toCat: [MLXArray] = []
        if trimSize > 0 {
            toCat = [
                array[.ellipsis, ..<keep, 0...],
                array[.ellipsis, (trimSize + keep)..., 0...],
            ]
        } else {
            toCat = [array]
        }
        if let append = append {
            toCat.append(append)
        }
        return concatenated(toCat, axis: 2)
    }

    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        // Rearrange the cache into temporal order, slicing off the end if unused
        if idx == array.dim(2) {
            return array
        } else if idx < offset {
            return concatenated(
                [
                    array[.ellipsis, ..<keep, 0...],
                    array[.ellipsis, idx..., 0...],
                    array[.ellipsis, keep ..< idx, 0...],
                ], axis: 2)
        } else {
            return array[.ellipsis, ..<idx, 0...]
        }
    }

    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            // For initial forward pass with long sequences, store all tokens
            // Sliding window should only be applied during generation phase
            self.keys = keys
            self.values = values
        } else {
            // Put the keys/values in temporal order to preserve context
            self.keys = temporalOrder(self.keys!)
            self.values = temporalOrder(self.values!)

            // The largest size is maxSize + S to ensure every token gets at least maxSize context
            let trimSize = idx - (maxCacheSize ?? idx)
            self.keys = trim(trimSize: trimSize, self.keys!, append: keys)
            self.values = trim(trimSize: trimSize, self.values!, append: values)
        }
        // Update offset and idx based on actual processing
        let actualTokensStored = self.keys!.dim(2)
        offset += keys.dim(2)  // Still track total tokens processed
        idx = actualTokensStored  // But idx reflects actual cache size

        return (self.keys!, self.values!)
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let S = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = offset

        // May not have hit the max size yet, so potentially keep growing the cache
        if self.keys == nil
            || (prev >= self.keys!.dim(2) && self.keys!.dim(2) < (maxCacheSize ?? Int.max))
        {
            let newSize: Int
            if let maxCacheSize = maxCacheSize {
                newSize = min(step, maxCacheSize - prev)
            } else {
                newSize = step  // If no max size, just grow by step size
            }

            let kShape = [B, nKVHeads, newSize, kHeadDim]
            let vShape = [B, nKVHeads, newSize, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
            idx = prev
        }

        // Trim if needed
        if let maxCacheSize = maxCacheSize {
            let trimSize = self.keys!.dim(2) - maxCacheSize
            if trimSize > 0 {
                self.keys = trim(trimSize: trimSize, self.keys!)
                self.values = trim(trimSize: trimSize, self.values!)
                idx = maxCacheSize
            }
        }

        // Rotate
        if let maxCacheSize = maxCacheSize, idx == maxCacheSize {
            idx = keep
        }

        // Assign
        self.keys![.ellipsis, idx ..< (idx + S), 0...] = keys
        self.values![.ellipsis, idx ..< (idx + S), 0...] = values
        offset += S
        idx += S

        // If the buffer is not full, slice off the end
        if let maxCacheSize = maxCacheSize, offset < maxCacheSize {
            return (
                self.keys![.ellipsis, ..<offset, 0...],
                self.values![.ellipsis, ..<offset, 0...]
            )
        }
        return (self.keys!, self.values!)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if keys.dim(2) == 1 {
            return updateInPlace(keys: keys, values: values)
        }
        return updateConcat(keys: keys, values: values)
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if offset < keys.dim(2) {
                return [
                    keys[.ellipsis, ..<offset, 0...],
                    values[.ellipsis, ..<offset, 0...],
                ]
            } else {
                return [keys, values]
            }
        }
        set {
            guard newValue.count == 2 else {
                fatalError("RotatingKVCache state must have exactly 2 arrays")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            // Note: RotatingKVCache doesn't set offset from keys like KVCache does
            // The offset is managed through meta_state
        }
    }

    public override var metaState: [String] {
        get {
            // Use "None" string for nil values to match Python behavior
            let maxSizeStr = maxCacheSize?.description ?? "None"
            return [String(keep), maxSizeStr, String(step), String(offset), String(idx)]
        }
        set {
            guard newValue.count == 5 else {
                fatalError("RotatingKVCache metaState must have exactly 5 values")
            }
            self.keep = Int(newValue[0]) ?? 0
            if newValue[1] == "None" {
                self.maxCacheSize = nil
            }
            self.step = Int(newValue[2]) ?? 256
            self.offset = Int(newValue[3]) ?? 0
            self.idx = Int(newValue[4]) ?? 0
        }
    }

    public override var isTrimmable: Bool {
        guard let maxCacheSize = maxCacheSize else { return true }
        return offset < maxCacheSize
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        idx -= trimmed
        return trimmed
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), maxSize: \(maxCacheSize?.description ?? "nil"), keep: \(keep), idx: \(idx)"
    }

    /// Convert to quantized cache
    /// Note: This is complex due to the rotating nature and temporal ordering
    public func toQuantized(groupSize: Int = 64, bits: Int = 4) -> QuantizedKVCache {
        // For now, throw an error like the Python version does
        // A full implementation would need to handle the temporal ordering correctly
        fatalError(
            "RotatingKVCache quantization not yet implemented - temporal ordering makes this complex"
        )

        // Future implementation would need to:
        // 1. Put keys/values in temporal order using temporalOrder()
        // 2. Quantize the temporally ordered arrays
        // 3. Store metadata about rotation state
        // 4. Implement corresponding dequantization with rotation restoration
    }
}

/// Quantized KV cache for memory efficiency using MLX quantization
public class QuantizedKVCache: BaseKVCache, QuantizedKVCacheProtocol {
    private var keys: (MLXArray, MLXArray, MLXArray)?
    private var values: (MLXArray, MLXArray, MLXArray)?
    // TODO: `offset` from the Python implementation is not implemented here. Do we need it?
    private let step: Int
    public let groupSize: Int
    public let bits: Int

    public init(groupSize: Int = 64, bits: Int = 8) {
        self.groupSize = groupSize
        self.bits = bits
        self.step = 256
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let keys = keys {
            arrays.append(contentsOf: [keys.0, keys.1, keys.2])
        }
        if let values = values {
            arrays.append(contentsOf: [values.0, values.1, values.2])
        }
        return arrays
    }

    /// Tree map equivalent for applying function to tuple elements
    private func treeMap<T>(_ transform: (MLXArray) -> T, _ tuple: (MLXArray, MLXArray, MLXArray))
        -> (T, T, T)
    {
        return (transform(tuple.0), transform(tuple.1), transform(tuple.2))
    }

    /// Tree map for two tuples (like Python's tree_map over (keys, values))
    private func treeMapPair<T>(
        _ transform: (MLXArray) -> T, _ tuple1: (MLXArray, MLXArray, MLXArray),
        _ tuple2: (MLXArray, MLXArray, MLXArray)
    ) -> ((T, T, T), (T, T, T)) {
        return (treeMap(transform, tuple1), treeMap(transform, tuple2))
    }

    /// Create initial quantized tuples (like Python's init_quant)
    private func initQuant(dim: Int, shape: [Int], dtype: DType) -> (MLXArray, MLXArray, MLXArray) {
        // Create temporary zero arrays and quantize them using native MLX Swift
        let tempArray = MLXArray.zeros(shape + [dim], dtype: dtype)
        let quantized = quantized(tempArray, groupSize: groupSize, bits: bits)

        return (quantized.wq, quantized.scales, quantized.biases)
    }

    /// Expand quantized tuple
    private func expandQuant(_ quantTuple: (MLXArray, MLXArray, MLXArray), newShape: [Int]) -> (
        MLXArray, MLXArray, MLXArray
    ) {
        return treeMap(
            { array in
                let newArray = MLXArray.zeros(newShape + [array.dim(-1)], dtype: array.dtype)
                return concatenated([array, newArray], axis: -2)
            }, quantTuple)
    }

    /// Get current quantized keys and values as tuples (efficient access)
    /// - Returns: Tuple of ((keyWeight, keyScales, keyBiases), (valueWeight, valueScales, valueBiases))
    public func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray), (MLXArray, MLXArray, MLXArray)
    )? {
        guard let keys = keys, let values = values else { return nil }

        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)

        return (trimmedKeys, trimmedValues)
    }

    /// Update cache and return quantized tuples (Python's update_and_fetch)
    /// This is needed because `update` in Swift must return `(MLXArray, MLXArray)`
    ///
    /// - Parameters:
    ///   - keys: New key data to add to cache
    ///   - values: New value data to add to cache
    /// - Returns: Quantized tuples (keys, values) as ((weight, scales, biases), (weight, scales, biases))
    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray), (MLXArray, MLXArray, MLXArray)
    ) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let numSteps = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = offset

        // Check if we need to expand the cache
        if self.keys == nil || (prev + numSteps) > self.keys!.0.dim(-2) {
            let newSteps = ((step + numSteps - 1) / step) * step
            let shape = [B, nKVHeads, newSteps]

            if let existingKeys = self.keys, let existingValues = self.values {
                // Trim if needed
                if prev % step != 0 {
                    // Use tree_map equivalent to trim both keys and values
                    let (trimmedKeys, trimmedValues) = treeMapPair(
                        { array in
                            array[.ellipsis, ..<prev, 0...]
                        }, existingKeys, existingValues)

                    self.keys = trimmedKeys
                    self.values = trimmedValues
                }

                // Expand using tree_map equivalent (Python's tree_map(expand_quant, ...))
                self.keys = expandQuant(self.keys!, newShape: shape)
                self.values = expandQuant(self.values!, newShape: shape)
            } else {
                // Initialize new quantized cache
                self.keys = initQuant(dim: kHeadDim, shape: shape, dtype: keys.dtype)
                self.values = initQuant(dim: vHeadDim, shape: shape, dtype: keys.dtype)
            }
        }

        offset += numSteps

        let quantizedKeys = quantized(keys, groupSize: groupSize, bits: bits)
        let quantizedValues = quantized(values, groupSize: groupSize, bits: bits)

        // Convert named tuples to positional tuples
        let qKeys = (quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases)
        let qValues = (quantizedValues.wq, quantizedValues.scales, quantizedValues.biases)

        // Assign to storage
        guard let currentKeys = self.keys, let currentValues = self.values else {
            fatalError("Quantized cache not properly initialized")
        }

        // Update each component of the quantized tuples
        currentKeys.0[.ellipsis, prev ..< offset, 0...] = qKeys.0
        currentKeys.1[.ellipsis, prev ..< offset, 0...] = qKeys.1
        currentKeys.2[.ellipsis, prev ..< offset, 0...] = qKeys.2

        currentValues.0[.ellipsis, prev ..< offset, 0...] = qValues.0
        currentValues.1[.ellipsis, prev ..< offset, 0...] = qValues.1
        currentValues.2[.ellipsis, prev ..< offset, 0...] = qValues.2

        self.keys = currentKeys
        self.values = currentValues

        // Return quantized tuples
        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentKeys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentValues)

        return (trimmedKeys, trimmedValues)
    }

    /// This method is required by the KVCache protocol, but it is not intended to be used with QuantizedKVCache.
    /// Use `updateQuantized` instead.
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "`update` was called on `QuantizedKVCache`. Use `updateQuantized` instead."
        )
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = keys, let values = values else { return [] }

            if offset < keys.0.dim(2) {
                // Trim to current offset using tree_map
                let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
                let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)
                // Flatten tuples to array for serialization
                return [
                    trimmedKeys.0, trimmedKeys.1, trimmedKeys.2, trimmedValues.0, trimmedValues.1,
                    trimmedValues.2,
                ]
            } else {
                // Flatten tuples to array for serialization
                return [keys.0, keys.1, keys.2, values.0, values.1, values.2]
            }
        }
        set {
            guard newValue.count == 6 else {
                fatalError(
                    "QuantizedKVCache state must have exactly 6 arrays (3 for keys, 3 for values)")
            }

            // Reconstruct tuples from flat array
            keys = (newValue[0], newValue[1], newValue[2])
            values = (newValue[3], newValue[4], newValue[5])
        }
    }

    public override var metaState: [String] {
        get { [String(step), String(offset), String(groupSize), String(bits)] }
        set {
            guard newValue.count == 4 else {
                fatalError("QuantizedKVCache metaState must have exactly 4 values")
            }

            self.offset = Int(newValue[1]) ?? 0

            // Validate that step, groupSize, and bits match current instance
            let expectedStep = Int(newValue[0]) ?? 256
            let expectedGroupSize = Int(newValue[2]) ?? 64
            let expectedBits = Int(newValue[3]) ?? 8

            if expectedStep != step || expectedGroupSize != groupSize || expectedBits != bits {
                print(
                    "Warning: QuantizedKVCache metaState mismatch - loaded cache has different parameters"
                )
                print("Expected: step=\(step), groupSize=\(groupSize), bits=\(bits)")
                print(
                    "Loaded: step=\(expectedStep), groupSize=\(expectedGroupSize), bits=\(expectedBits)"
                )
            }
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    /// Convert to unquantized cache
    public func toUnquantized() -> KVCacheSimple {
        let simpleCache = KVCacheSimple()
        simpleCache.offset = self.offset

        if let keys = keys, let values = values {
            // Dequantize the current state using tree_map approach
            let currentKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
            let currentValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)

            let dequantizedKeys = dequantized(
                currentKeys.0, scales: currentKeys.1, biases: currentKeys.2,
                groupSize: groupSize, bits: bits)
            let dequantizedValues = dequantized(
                currentValues.0, scales: currentValues.1, biases: currentValues.2,
                groupSize: groupSize, bits: bits)

            // Set the unquantized state
            simpleCache.state = [dequantizedKeys, dequantizedValues]
        }

        return simpleCache
    }
}

/// Chunked KV cache for processing large contexts in chunks
public class ChunkedKVCache: KVCacheSimple {
    private var chunkSize: Int?
    private var startPosition: Int = 0

    public init(chunkSize: Int? = nil) {
        self.chunkSize = chunkSize
        super.init()
    }

    public func maybeTrimFront() {
        guard let keys = self.keys,
            let chunkSize = chunkSize,
            keys.dim(2) >= chunkSize
        else { return }

        startPosition += keys.dim(2) - chunkSize
        self.keys = keys[.ellipsis, (-chunkSize)..., 0...]
        self.values = values?[.ellipsis, (-chunkSize)..., 0...]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset - startPosition

        if self.keys == nil || (prev + keys.dim(2)) > self.keys!.dim(2) {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if prev % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<prev, 0...]
                    currentValues = currentValues[.ellipsis, ..<prev, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        offset += keys.dim(2)
        let end = offset - startPosition
        self.keys![.ellipsis, prev ..< end, 0...] = keys
        self.values![.ellipsis, prev ..< end, 0...] = values

        return (self.keys![.ellipsis, ..<end, 0...], self.values![.ellipsis, ..<end, 0...])
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset - startPosition, n)
        offset -= trimmed
        return trimmed
    }

    public override var metaState: [String] {
        get {
            // Use "None" string for nil values to match Python behavior
            let chunkSizeStr = chunkSize?.description ?? "None"
            return [chunkSizeStr, String(startPosition)]
        }
        set {
            guard newValue.count == 2 else {
                fatalError("ChunkedKVCache metaState must have exactly 2 values")
            }
            if newValue[0] == "None" {
                self.chunkSize = nil
            }
            self.startPosition = Int(newValue[1]) ?? 0
        }
    }
}

/// Simple cache for Mamba-style state space models
public class MambaCache: BaseKVCache {
    private var cache: [MLXArray?] = [nil, nil]

    public override init() {
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        cache.compactMap { $0 }
    }

    public subscript(index: Int) -> MLXArray? {
        get { cache[index] }
        set { cache[index] = newValue }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        // Mamba doesn't use traditional KV cache update pattern
        fatalError("MambaCache should not use update(keys:values:) - use subscript access instead")
    }

    public override var state: [MLXArray] {
        get {
            // Need to preserve the structure including nils, similar to Python version
            // Use empty arrays as placeholders for nil values
            var result: [MLXArray] = []
            for item in cache {
                if let array = item {
                    result.append(array)
                } else {
                    // Use an empty array as placeholder for nil (this shape should never occur naturally)
                    result.append(MLXArray.zeros([0], dtype: .float32))
                }
            }
            return result
        }
        set {
            guard newValue.count == cache.count else {
                fatalError("MambaCache state must have exactly \(cache.count) elements")
            }
            for (i, array) in newValue.enumerated() {
                // Check if this is our nil placeholder (empty array with size 0)
                if array.size == 0 {
                    cache[i] = nil
                } else {
                    cache[i] = array
                }
            }
        }
    }
}

/// Composite cache that manages multiple sub-caches
public class CacheList: BaseKVCache {
    private var caches: [KVCache]

    public init(_ caches: KVCache...) {
        self.caches = caches
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        caches.flatMap { $0.innerState() }
    }

    public subscript(index: Int) -> KVCache {
        return caches[index]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("CacheList should not use update(keys:values:) - use subscript access instead")
    }

    public override var state: [MLXArray] {
        get { caches.flatMap { $0.state } }
        set {
            let stateLengths = caches.map { $0.state.count }
            var start = 0
            for i in 0 ..< caches.count {
                let length = stateLengths[i]
                caches[i].state = Array(newValue[start ..< (start + length)])
                start += length
            }
        }
    }

    public override var isTrimmable: Bool {
        caches.allSatisfy { $0.isTrimmable }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        return caches.first?.trim(n) ?? 0
    }
}

// MARK: - Error Types

struct KVCacheError: Error {
    let message: String
}

// MARK: - Utility Functions

/// Save a pre-computed prompt cache to a file.
///
/// - Parameters:
///   - fileName: The `.safetensors` file name
///   - cache: The model cache state
///   - metadata: Optional metadata to save along with cache state
public func savePromptCache(
    fileName: String,
    cache: [KVCache],
    metadata: [String: String] = [:]
) throws {
    let cacheData = cache.map { $0.state }
    let cacheInfo = cache.map { $0.metaState }
    // Use Python-compatible class names for cross-platform compatibility
    let cacheClasses = cache.map { cache -> String in
        switch cache {
        case is KVCacheSimple:
            return "KVCache"  // Python uses "KVCache" for the basic cache
        case is RotatingKVCache:
            return "RotatingKVCache"
        case is QuantizedKVCache:
            return "QuantizedKVCache"
        case is ChunkedKVCache:
            return "ChunkedKVCache"
        case is MambaCache:
            return "MambaCache"
        case is CacheList:
            return "CacheList"
        default:
            return "KVCache"  // Default fallback
        }
    }

    // Flatten cache data into a dictionary with string keys
    var flattenedData: [String: MLXArray] = [:]
    for (i, arrays) in cacheData.enumerated() {
        for (j, array) in arrays.enumerated() {
            flattenedData["cache_\(i)_\(j)"] = array
        }
    }

    // Store metadata and cache structure info as special arrays
    // Since MLX Swift doesn't support metadata in safetensors, we encode it as arrays

    // Cache structure metadata
    flattenedData["__metadata_num_caches"] = MLXArray([cacheData.count])
    flattenedData["__metadata_cache_data_counts"] = MLXArray(cacheData.map { $0.count })
    flattenedData["__metadata_cache_info_counts"] = MLXArray(cacheInfo.map { $0.count })

    // Cache classes (encoded as integers to work around string limitation)
    let classMapping = [
        "KVCache": 0, "RotatingKVCache": 1, "QuantizedKVCache": 2,
        "ChunkedKVCache": 3, "MambaCache": 4, "CacheList": 5,
    ]
    let classIndices = cacheClasses.map { classMapping[$0] ?? 0 }
    flattenedData["__metadata_cache_classes"] = MLXArray(classIndices)

    // Cache info (flattened)
    var flatCacheInfo: [String] = []
    for info in cacheInfo {
        flatCacheInfo.append(contentsOf: info)
    }
    // Encode strings as UTF-8 bytes and store as arrays
    for (i, infoString) in flatCacheInfo.enumerated() {
        let bytes = Array(infoString.utf8)
        if !bytes.isEmpty {
            flattenedData["__metadata_cache_info_\(i)"] = MLXArray(bytes.map { Int32($0) })
        }
    }

    // User metadata
    for (i, (key, value)) in metadata.enumerated() {
        let keyBytes = Array(key.utf8)
        let valueBytes = Array(value.utf8)
        if !keyBytes.isEmpty {
            flattenedData["__metadata_user_key_\(i)"] = MLXArray(keyBytes.map { Int32($0) })
        }
        if !valueBytes.isEmpty {
            flattenedData["__metadata_user_value_\(i)"] = MLXArray(valueBytes.map { Int32($0) })
        }
    }
    flattenedData["__metadata_user_count"] = MLXArray([metadata.count])

    try save(arrays: flattenedData, url: URL(fileURLWithPath: fileName))
}

/// Load a prompt cache from a file.
///
/// - Parameters:
///   - fileName: The `.safetensors` file name
///   - returnMetadata: Whether to return metadata along with the cache
/// - Returns: The prompt cache and optionally the metadata
public func loadPromptCache(
    fileName: String,
    returnMetadata: Bool = false
) throws -> ([KVCache], [String: String]?) {
    let arrays = try loadArrays(url: URL(fileURLWithPath: fileName))

    // Extract metadata from special arrays
    guard let numCachesArray = arrays["__metadata_num_caches"],
        let dataCountsArray = arrays["__metadata_cache_data_counts"],
        let infoCountsArray = arrays["__metadata_cache_info_counts"],
        let classIndicesArray = arrays["__metadata_cache_classes"]
    else {
        throw KVCacheError(message: "Invalid cache file format - missing metadata")
    }

    let numCaches = numCachesArray.item(Int.self)
    let dataCounts = dataCountsArray.asArray(Int.self)
    let infoCounts = infoCountsArray.asArray(Int.self)
    let classIndices = classIndicesArray.asArray(Int.self)

    guard
        dataCounts.count == numCaches && infoCounts.count == numCaches
            && classIndices.count == numCaches
    else {
        throw KVCacheError(message: "Mismatch in cache counts")
    }

    // Reconstruct cache data
    var cacheData: [[MLXArray]] = []
    for i in 0 ..< numCaches {
        var cacheArrays: [MLXArray] = []
        for j in 0 ..< dataCounts[i] {
            if let array = arrays["cache_\(i)_\(j)"] {
                cacheArrays.append(array)
            }
        }
        cacheData.append(cacheArrays)
    }

    // Reconstruct cache info
    var cacheInfo: [[String]] = []
    var infoIndex = 0
    for i in 0 ..< numCaches {
        var info: [String] = []
        for _ in 0 ..< infoCounts[i] {
            if let infoArray = arrays["__metadata_cache_info_\(infoIndex)"] {
                let bytes = infoArray.asArray(Int32.self).map { UInt8($0) }
                let infoString = String(bytes: bytes, encoding: .utf8) ?? ""
                info.append(infoString)
            }
            infoIndex += 1
        }
        cacheInfo.append(info)
    }

    // Reconstruct cache classes
    let classMapping = [
        0: "KVCache", 1: "RotatingKVCache", 2: "QuantizedKVCache",
        3: "ChunkedKVCache", 4: "MambaCache", 5: "CacheList",
    ]

    // Reconstruct cache instances
    var caches: [KVCache] = []
    for i in 0 ..< numCaches {
        let className = classMapping[classIndices[i]] ?? "KVCache"

        var cache: KVCache
        switch className {
        case "KVCache", "KVCacheSimple":  // Handle both Python and Swift names
            cache = KVCacheSimple()
        case "RotatingKVCache":
            cache = RotatingKVCache(maxSize: nil)  // Will be set from metaState
        case "QuantizedKVCache":
            cache = QuantizedKVCache()
        case "ChunkedKVCache":
            cache = ChunkedKVCache()
        case "MambaCache":
            cache = MambaCache()
        case "CacheList":
            // Note: CacheList requires special handling as it contains sub-caches
            // For now, create an empty CacheList - this may not work correctly
            // for complex cache hierarchies loaded from Python
            cache = CacheList()
            print("Warning: CacheList loading may not preserve sub-cache structure correctly")
        default:
            throw KVCacheError(message: "Unknown cache class: \(className)")
        }

        cache.state = cacheData[i]
        cache.metaState = cacheInfo[i]
        caches.append(cache)
    }

    // Extract user metadata
    var userMetadata: [String: String] = [:]
    if let userCountArray = arrays["__metadata_user_count"] {
        let userCount = userCountArray.item(Int.self)
        for i in 0 ..< userCount {
            var key = ""
            var value = ""

            if let keyArray = arrays["__metadata_user_key_\(i)"] {
                let keyBytes = keyArray.asArray(Int32.self).map { UInt8($0) }
                key = String(bytes: keyBytes, encoding: .utf8) ?? ""
            }

            if let valueArray = arrays["__metadata_user_value_\(i)"] {
                let valueBytes = valueArray.asArray(Int32.self).map { UInt8($0) }
                value = String(bytes: valueBytes, encoding: .utf8) ?? ""
            }

            if !key.isEmpty {
                userMetadata[key] = value
            }
        }
    }

    if returnMetadata {
        return (caches, userMetadata)
    }
    return (caches, nil)
}

/// Construct the model's cache for use when generating.
///
/// This function will defer the cache construction to the model if it has a
/// `newCache` method, otherwise it will make a default KV cache.
public func makePromptCache(
    model: any LanguageModel,
    parameters: GenerateParameters? = nil
) -> [KVCache] {
    // The model already conforms to LanguageModel which has newCache
    // If it also conforms to KVCacheDimensionProvider, the extension will provide the implementation
    return model.newCache(parameters: parameters)
}

/// Legacy function for backwards compatibility
public func makePromptCache(
    model: any LanguageModel,
    maxKVSize: Int? = nil
) -> [KVCache] {
    let parameters = maxKVSize.map { GenerateParameters(maxKVSize: $0) }
    return makePromptCache(model: model, parameters: parameters)
}

/// Fallback function to create cache when layer count is known
///
/// This function creates a default cache structure when the number of layers is known.
/// Use this when `makePromptCache` cannot determine the layer count automatically.
public func makePromptCacheWithLayerCount(
    numLayers: Int,
    maxKVSize: Int? = nil
) -> [KVCache] {
    if let maxKVSize = maxKVSize {
        return (0 ..< numLayers).map { _ in
            RotatingKVCache(maxSize: maxKVSize, keep: 4)
        }
    } else {
        return (0 ..< numLayers).map { _ in KVCacheSimple() }
    }
}

/// Check if model's cache can be trimmed.
public func canTrimPromptCache(_ cache: [KVCache]) -> Bool {
    return cache.allSatisfy { $0.isTrimmable }
}

/// Trim the model's cache by the given number of tokens.
///
/// This function will trim the cache if possible (in-place) and return the
/// number of tokens that were trimmed.
@discardableResult
public func trimPromptCache(_ cache: [KVCache], numTokens: Int) -> Int {
    guard canTrimPromptCache(cache), !cache.isEmpty else { return 0 }
    return cache.first?.trim(numTokens) ?? 0
}

// MARK: - Type Aliases

/// Standard KV cache - alias to KVCacheSimple for compatibility
public typealias StandardKVCache = KVCacheSimple

// MARK: - Quantized Attention Operations

public func quantizedScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray),
    quantizedValues: (MLXArray, MLXArray, MLXArray),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    groupSize: Int = 64,
    bits: Int = 8
) -> MLXArray {

    let (B, nQHeads, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
    let nKVHeads = quantizedKeys.0.dim(-3)
    let nRepeats = nQHeads / nKVHeads

    // Scale queries
    var scaledQueries = queries * scale

    // Handle GQA (Grouped Query Attention)
    var qKeys = quantizedKeys
    var qValues = quantizedValues
    if nRepeats > 1 {
        scaledQueries = scaledQueries.reshaped([B, nKVHeads, nRepeats, L, D])
        qKeys = (
            expandedDimensions(qKeys.0, axis: -3),
            expandedDimensions(qKeys.1, axis: -3),
            expandedDimensions(qKeys.2, axis: -3)
        )
        qValues = (
            expandedDimensions(qValues.0, axis: -3),
            expandedDimensions(qValues.1, axis: -3),
            expandedDimensions(qValues.2, axis: -3)
        )
    }

    // Compute attention scores using quantized matmul
    var scores = quantizedMatmul(
        scaledQueries, qKeys.0, scales: qKeys.1, biases: qKeys.2,
        transpose: true, groupSize: groupSize, bits: bits
    )

    // Apply mask
    switch mask {
    case .causal:
        let (qL, kL) = (scores.dim(-2), scores.dim(-1))
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0 ..< kL)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1), expandedDimensions(kIndices, axis: -2))
        scores = MLX.where(causalMask, scores, MLXArray(Float.leastNormalMagnitude))

    case .array(let maskArray):
        if maskArray.dtype == .bool {
            scores = MLX.where(maskArray, scores, MLXArray(Float.leastNormalMagnitude))
        } else {
            scores = scores + maskArray
        }

    case .arrays(let maskArrays):
        // Handle multiple mask arrays - just use the first one for simplicity
        if let maskArray = maskArrays.first {
            if maskArray.dtype == .bool {
                scores = MLX.where(maskArray, scores, MLXArray(Float.leastNormalMagnitude))
            } else {
                scores = scores + maskArray
            }
        }

    case .none:
        break
    }

    let attentionWeights = softmax(scores, axis: -1)

    // Compute output using quantized matmul
    var output = quantizedMatmul(
        attentionWeights, qValues.0, scales: qValues.1, biases: qValues.2,
        transpose: false, groupSize: groupSize, bits: bits
    )

    // Reshape output for GQA
    if nRepeats > 1 {
        output = output.reshaped([B, nQHeads, L, D])
    }

    return output
}

// MARK: - Dynamic Cache Quantization

/// Dynamically quantize KV caches during generation if conditions are met
///
/// Converts regular caches to quantized caches when:
/// - kvBits is specified
/// - The cache is not already quantized
/// - The cache offset is greater than quantizedKVStart
///
/// - Parameters:
///   - cache: Array of KV caches to potentially quantize
///   - kvBits: Number of bits for quantization (nil = no quantization)
///   - kvGroupSize: Group size for quantization
///   - quantizedKVStart: Step to begin quantizing
public func maybeQuantizeKVCache(
    cache: inout [KVCache],
    kvBits: Int?,
    kvGroupSize: Int = 64,
    quantizedKVStart: Int = 0
) {
    guard let kvBits = kvBits,
        !cache.isEmpty,
        !(cache[0] is QuantizedKVCache),
        cache[0].offset > quantizedKVStart
    else {
        return
    }

    for i in 0 ..< cache.count {
        if let simpleCache = cache[i] as? KVCacheSimple {
            cache[i] = simpleCache.toQuantized(groupSize: kvGroupSize, bits: kvBits)
        }
        // Note: RotatingKVCache.toQuantized() is not implemented yet like in Python
        // When implemented, add: else if let rotatingCache = cache[i] as? RotatingKVCache { ... }
    }
}
