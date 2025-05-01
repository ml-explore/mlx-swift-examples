// Copyright © 2024 Apple Inc.

import Foundation
import MLX

// MARK: - KVCache Protocol

/// Interface for Key/Value cache for LLMs.
public protocol KVCache {
    /// Get the current offset (total number of tokens processed).
    var offset: Int { get }

    /// Update the cache with new keys and values and return the full cached keys and values.
    /// - Parameters:
    ///   - keys: New keys tensor, typically shape [Batch, Heads, SeqLen, HeadDim]
    ///   - values: New values tensor, typically shape [Batch, Heads, SeqLen, HeadDim]
    /// - Returns: A tuple containing the updated full keys and values tensors up to the current offset.
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// Check if the cache can be trimmed (typically true for standard/quantized, conditional for rotating).
    func isTrimmable() -> Bool

    /// Trim the cache state by `count` tokens from the end.
    /// Does not shrink allocated memory, only adjusts the logical size (`offset`).
    /// - Parameter count: The number of tokens to trim.
    /// - Returns: The actual number of tokens trimmed (capped by current `offset`).
    func trim(count: Int) -> Int

    // Note: Saving/Loading state requires separate handling, potentially outside the protocol
    // or via specific methods if a unified approach is desired.
    // The Python version uses external functions accessing `state` and `meta_state` properties.
}

// MARK: - Standard Concatenating KVCache

/// A KVCache implementation that concatenates new keys and values along the sequence dimension.
/// Suitable for layers attending to the full sequence history.
public class StandardKVCache: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    private var currentCapacity: Int = 0  // Track allocated sequence length

    public private(set) var offset = 0
    let step: Int  // Resizing step size

    public init(step: Int = 256) {
        self.step = step
    }

    // Internal state for potential saving (mimics Python's `state` property)
    public var state: (keys: MLXArray?, values: MLXArray?) {
        get {
            if let k = keys, let v = values, offset < currentCapacity {
                // Return only the valid portion
                return (k[0..., 0..., ..<offset, 0...], v[0..., 0..., ..<offset, 0...])
            } else {
                // Return full arrays if offset matches capacity or if nil
                return (keys, values)
            }
        }
        set {
            self.keys = newValue.keys
            self.values = newValue.values
            self.offset = self.keys?.dim(2) ?? 0
            self.currentCapacity = self.offset  // Assume loaded state is exactly the right size
        }
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let previousOffset = self.offset
        let newSeqLen = newKeys.dim(2)  // Assuming [B, H, L, D]

        // Check if resizing is needed
        let requiredCapacity = previousOffset + newSeqLen
        let needsResize = keys == nil || requiredCapacity > currentCapacity

        if needsResize {
            let B = newKeys.dim(0)
            let kvHeads = newKeys.dim(1)
            let kHeadDim = newKeys.dim(3)
            let vHeadDim = newValues.dim(3)

            // Calculate new capacity based on steps
            // Ensure enough space: round up requiredCapacity to the nearest multiple of step
            let nSteps = (requiredCapacity + step - 1) / step
            let newCapacity = nSteps * step

            let kShape = [B, kvHeads, newCapacity, kHeadDim]
            let vShape = [B, kvHeads, newCapacity, vHeadDim]

            // Use `zeros` which might be slightly less efficient than `empty` if available,
            // but ensures initialized memory.
            let resizedK = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            let resizedV = MLXArray.zeros(vShape, dtype: newValues.dtype)

            // Copy existing data if it exists
            if let currentKeys = self.keys, let currentValues = self.values, previousOffset > 0 {
                // Ensure we only copy the valid part of the old cache
                resizedK[0..., 0..., ..<previousOffset, 0...] =
                    currentKeys[0..., 0..., ..<previousOffset, 0...]
                resizedV[0..., 0..., ..<previousOffset, 0...] =
                    currentValues[0..., 0..., ..<previousOffset, 0...]
            }
            self.keys = resizedK
            self.values = resizedV
            self.currentCapacity = newCapacity
        }

        // Insert the new keys/values into the (potentially resized) cache
        // Use non-optional keys/values now, guarded by the resize logic above.
        self.keys![0..., 0..., previousOffset ..< requiredCapacity, 0...] = newKeys
        self.values![0..., 0..., previousOffset ..< requiredCapacity, 0...] = newValues

        // Update offset
        self.offset = requiredCapacity

        // Return the valid portion of the cache up to the new offset
        return (
            self.keys![0..., 0..., ..<self.offset, 0...],
            self.values![0..., 0..., ..<self.offset, 0...]
        )
    }

    public func isTrimmable() -> Bool {
        true  // Standard cache can always be logically trimmed
    }

    public func trim(count: Int) -> Int {
        let trimmedCount = min(self.offset, count)
        self.offset -= trimmedCount
        return trimmedCount
    }

    // public func toQuantized(groupSize: Int = 64, bits: Int = 4) -> QuantizedKVCache {
    //     // Implementation depends on QuantizedKVCache and MLX Swift quantization API
    //     fatalError("Not implemented")
    // }
}

// MARK: - Rotating KVCache

/// A KVCache implementation that uses a fixed-size buffer (`maxSize`) and rotates entries,
/// keeping the first `keep` tokens fixed. Mimics the Python MLX RotatingKVCache.
public class RotatingKVCache: KVCache {

    private var keys: MLXArray?
    private var values: MLXArray?

    // MARK: Metadata State (Restored via load method)
    public private(set) var offset = 0  // Total tokens processed logically
    private var idx = 0  // Current insertion index within the buffer
    private var currentSize = 0  // Current allocated buffer size (up to maxSize)

    // MARK: Configuration (Set at init)
    let maxSize: Int
    let keep: Int  // Number of initial tokens to always keep
    let step: Int  // Growth step size when currentSize < maxSize

    /// Initializes a RotatingKVCache.
    /// - Parameters:
    ///   - maxSize: The maximum sequence length to store (the sliding window size). Must be > 0.
    ///   - keep: The number of initial tokens to always keep (must be >= 0 and <= maxSize).
    ///   - step: The allocation growth step size.
    public init(maxSize: Int, keep: Int, step: Int = 256) {
        precondition(keep >= 0, "keep must be non-negative")
        precondition(maxSize > 0, "maxSize must be positive")
        precondition(keep <= maxSize, "keep must be less than or equal to maxSize")
        self.maxSize = maxSize
        self.keep = keep
        self.step = step
        self.idx = keep  // Initial insertion point is after the kept tokens (or 0 if keep is 0)
    }

    // MARK: State Properties (for Saving/Loading)

    /// Gets the current state (arrays) suitable for saving.
    /// Returns slices of the internal buffer up to the current logical offset.
    /// Matches Python's behavior: returns full arrays only if offset == buffer dimension.
    public var state: (keys: MLXArray?, values: MLXArray?) {
        guard let k = keys, let v = values else { return (nil, nil) }

        // Check if the logical offset exactly matches the allocated buffer dimension
        if offset == k.dim(2) && offset == v.dim(2) {
            return (k, v)  // Return full arrays
        } else {
            // Otherwise, return slices up to the logical offset
            // Ensure offset doesn't exceed buffer dimensions for slicing
            let sliceOffset = min(offset, k.dim(2), v.dim(2))
            return (
                k[0..., 0..., ..<sliceOffset, 0...],
                v[0..., 0..., ..<sliceOffset, 0...]
            )
        }
        // REMOVED Setter: State must be restored using the `load` method with metadata.
        // set { ... } // DO NOT USE - leads to incorrect state restoration
    }

    /// Gets the metadata required for proper state restoration.
    public var metaState:
        (keep: Int, maxSize: Int, step: Int, offset: Int, idx: Int, currentSize: Int)
    {
        (keep, maxSize, step, offset, idx, currentSize)
    }

    /// Loads the cache state from saved arrays and metadata.
    /// This is the **correct** way to restore state for RotatingKVCache.
    /// - Parameters:
    ///   - state: The tuple containing the key and value arrays (as saved by the `state` getter).
    ///   - metaState: The tuple containing the saved metadata.
    /// - Throws: An error if the loaded metadata's configuration (keep, maxSize, step) doesn't match the current instance.
    public func load(
        state: (keys: MLXArray?, values: MLXArray?),
        metaState: (keep: Int, maxSize: Int, step: Int, offset: Int, idx: Int, currentSize: Int)
    ) throws {
        // --- Validate Configuration ---
        guard metaState.keep == self.keep,
            metaState.maxSize == self.maxSize,
            metaState.step == self.step
        else {
            throw NSError(
                domain: "KVCacheError", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Metadata mismatch during RotatingKVCache state loading. Expected (keep: \(self.keep), maxSize: \(self.maxSize), step: \(self.step)), got (keep: \(metaState.keep), maxSize: \(metaState.maxSize), step: \(metaState.step))"
                ])
        }

        // --- Restore Metadata ---
        self.offset = metaState.offset
        self.idx = metaState.idx
        self.currentSize = metaState.currentSize  // Restore the known allocated size

        // --- Restore Arrays ---
        self.keys = state.keys
        self.values = state.values

        // --- Optional: Validation Checks ---
        if let k = self.keys, k.dim(2) != self.currentSize {
            print(
                "Warning: Loaded keys dimension (\(k.dim(2))) doesn't match restored currentSize (\(self.currentSize))"
            )
            // Depending on requirements, you might adjust currentSize or throw an error here.
            // For now, we trust the loaded metaState's currentSize.
        }
        if let v = self.values, v.dim(2) != self.currentSize {
            print(
                "Warning: Loaded values dimension (\(v.dim(2))) doesn't match restored currentSize (\(self.currentSize))"
            )
        }

        print(
            "RotatingKVCache state loaded using metadata: offset=\(self.offset), idx=\(self.idx), currentSize=\(self.currentSize)"
        )
    }

    // MARK: KVCache Protocol Methods

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let S = newKeys.dim(2)  // Sequence length of the update

        // Use efficient in-place update for single tokens (common during generation)
        if S == 1 {
            return updateInPlace(keys: newKeys, values: newValues)
        } else {
            // Use concatenation for multi-token updates (less common, e.g., prefill)
            return updateConcat(keys: newKeys, values: newValues)
        }
    }

    public func isTrimmable() -> Bool {
        // Match Python's definition: Trimmable only if not yet full
        return self.offset < self.maxSize
    }

    /// Trims the cache state logically by reducing the offset and adjusting the insertion index.
    /// Does not shrink allocated memory. Matches Python's behavior.
    /// - Parameter count: The number of tokens to trim from the end.
    /// - Returns: The actual number of tokens trimmed.
    public func trim(count: Int) -> Int {
        let trimmedCount = min(self.offset, count)  // Don't trim more than available
        self.offset -= trimmedCount
        // ** tärkeä (important) ** Adjust idx as well, matching Python
        self.idx -= trimmedCount
        // Note: Python doesn't clamp idx here. If trimming makes idx < keep,
        // the next insertion logic might need careful review, but let's match Python first.
        return trimmedCount
    }

    // MARK: - Internal Update Logic

    // Handles single-token updates (efficient in-place rotation)
    private func updateInPlace(keys newKey: MLXArray, values newValue: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let S = 1  // Single token update

        // 1. Grow buffer if needed and not yet at max size
        // Check if current buffer is full *logically* (offset >= currentSize)
        // AND if the buffer *physically* hasn't reached maxSize
        let needsGrow = (keys == nil) || (offset >= currentSize && currentSize < maxSize)

        if needsGrow {
            let B = newKey.dim(0)
            let kvHeads = newKey.dim(1)
            let kHeadDim = newKey.dim(3)
            let vHeadDim = newValue.dim(3)

            // Determine growth size: grow by step, but don't exceed maxSize
            let growth = min(step, maxSize - currentSize)
            let newBufferSeqLen = currentSize + growth

            let kShape = [B, kvHeads, newBufferSeqLen, kHeadDim]
            let vShape = [B, kvHeads, newBufferSeqLen, vHeadDim]

            let grownK = MLXArray.zeros(kShape, dtype: newKey.dtype)
            let grownV = MLXArray.zeros(vShape, dtype: newValue.dtype)

            // Copy existing data if it exists
            if let currentKeys = self.keys, let currentValues = self.values, currentSize > 0 {
                // Copy the entire old buffer
                grownK[0..., 0..., ..<currentSize, 0...] = currentKeys
                grownV[0..., 0..., ..<currentSize, 0...] = currentValues
            }
            self.keys = grownK
            self.values = grownV
            // Update currentSize to reflect new allocated size
            self.currentSize = newBufferSeqLen
            // After growing, idx should point to the first new slot, which is the old size
            self.idx = offset  // Python sets self._idx = prev (which is offset here)
        }

        // Ensure keys/values exist now (should be guaranteed by growth logic or previous state)
        guard var currentKeys = self.keys, var currentValues = self.values else {
            fatalError("Cache arrays are unexpectedly nil after growth check.")
        }

        // --- Python's _update_in_place logic adapted ---

        // Python first checks for trimming *if* the buffer is already at max_size
        // This seems slightly out of order compared to insertion, but let's try matching.
        // However, Python's trim logic (`_trim`) isn't called here directly.
        // Let's stick to the rotation and insertion logic first.

        // 2. Check if rotation within the buffer is needed (wrap idx)
        // This happens when the insertion index `idx` reaches the end of the *full* buffer.
        if idx == maxSize {  // Check against maxSize, not currentSize
            idx = keep  // Wrap around insertion point to after the kept tokens
        }

        // 3. Insert the new token at the current index `idx`
        // Ensure idx is within the bounds of the current allocated size
        if idx < currentSize {
            currentKeys[0..., 0..., idx ..< (idx + S), 0...] = newKey
            currentValues[0..., 0..., idx ..< (idx + S), 0...] = newValue
        } else {
            // This case should ideally not happen if growth/rotation logic is correct
            // It might indicate idx went out of bounds.
            fatalError(
                "Cache insertion index \(idx) is out of bounds for current size \(currentSize). Offset=\(offset)"
            )
        }

        // 4. Update pointers
        offset += S
        idx += S  // Move insertion pointer forward

        self.keys = currentKeys  // Update struct's stored arrays
        self.values = currentValues

        // 5. Return the relevant cache view
        // Python returns slice up to offset if offset < max_size, else full buffer.
        if offset < maxSize {
            // Slice up to the new logical offset, ensuring not to exceed buffer bounds
            let sliceOffset = min(offset, currentSize)
            return (
                currentKeys[0..., 0..., ..<sliceOffset, 0...],
                currentValues[0..., 0..., ..<sliceOffset, 0...]
            )
        } else {
            // Return the full buffer once logical offset reaches or exceeds max_size
            return (currentKeys, currentValues)
        }
    }

    // Handles multi-token updates (less efficient, involves reordering/concatenation)
    // This implementation attempts to closely follow Python's _update_concat and _trim
    private func updateConcat(keys newKeys: MLXArray, values newValues: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let S = newKeys.dim(2)

        guard var currentKeys = self.keys, var currentValues = self.values else {
            // First update: Just store the new keys/values
            self.keys = newKeys
            self.values = newValues
            self.offset = S
            self.currentSize = S  // Initial size is S
            self.idx = S  // Insertion point is at the end
            // Ensure size doesn't exceed maxSize even on first update
            if self.currentSize > self.maxSize {
                print(
                    "Warning: Initial prefill size (\(S)) exceeds maxSize (\(maxSize)). Truncating."
                )
                self.keys = self.keys?[0..., 0..., ..<self.maxSize, 0...]
                self.values = self.values?[0..., 0..., ..<self.maxSize, 0...]
                self.offset = self.maxSize
                self.currentSize = self.maxSize
                self.idx = self.maxSize
            }
            return (self.keys!, self.values!)
        }

        // --- Python's _update_concat logic adapted ---

        // 1. Put current cache in temporal order (using helper)
        // Note: This creates copies and can be expensive.
        currentKeys = temporalOrder(currentKeys)
        currentValues = temporalOrder(currentValues)

        // 2. Calculate trim size based on Python's logic: trim_size = self._idx - self.max_size
        // Here, _idx corresponds to the buffer size *before* temporal ordering.
        // Let's use currentSize (which should reflect the buffer size before this update).
        // Python's `_idx` in `_update_concat` seems to refer to the size *before* temporal ordering.
        let trimSize = max(0, currentSize - maxSize)  // How many elements to remove from the middle

        // 3. Trim the middle part (if needed) using a helper similar to Python's _trim
        currentKeys = trimMiddle(trimSize: trimSize, v: currentKeys)
        currentValues = trimMiddle(trimSize: trimSize, v: currentValues)

        // 4. Concatenate new keys/values
        self.keys = MLX.concatenated([currentKeys, newKeys], axis: 2)
        self.values = MLX.concatenated([currentValues, newValues], axis: 2)

        // 5. Update pointers
        self.offset += S
        self.currentSize = self.keys!.dim(2)  // Update current size after concat
        self.idx = self.currentSize  // After concat, idx points to the end

        // Ensure final size doesn't exceed maxSize after concatenation
        if self.currentSize > self.maxSize {
            print(
                "Warning: Concatenated size (\(self.currentSize)) exceeds maxSize (\(maxSize)). Truncating."
            )
            self.keys = self.keys?[0..., 0..., ..<self.maxSize, 0...]
            self.values = self.values?[0..., 0..., ..<self.maxSize, 0...]
            // Adjust pointers after truncation
            self.offset = min(self.offset, self.maxSize)  // Cap logical offset too
            self.currentSize = self.maxSize
            self.idx = self.maxSize
        }

        return (self.keys!, self.values!)
    }

    // MARK: - Helper Functions (Internal)

    // Helper to rearrange the buffer into temporal order
    // Note: This creates copies and can be expensive.
    private func temporalOrder(_ v: MLXArray) -> MLXArray {
        let bufferSeqLen = v.dim(2)  // Use actual dimension
        let logicalOffset = self.offset  // Use logical offset for slicing end
        let insertionIdx = self.idx  // Use internal index for rotation logic

        // Slice index should not exceed buffer length
        let effectiveOffset = min(logicalOffset, bufferSeqLen)

        // Case 1: No rotation needed or buffer not full yet
        // Python checks `self._idx == v.shape[2]` OR `self._idx >= self.offset` (implicit in else)
        if insertionIdx == bufferSeqLen || insertionIdx >= logicalOffset || bufferSeqLen <= keep {
            // Already in order, or only contains kept tokens, or insertion is beyond logical offset
            return v[0..., 0..., ..<effectiveOffset, 0...]
        }
        // Case 2: Rotation has occurred (idx < logicalOffset and bufferSeqLen > keep)
        else {
            // Buffer: [ Keep | Rotated Part 2 (oldest) | Rotated Part 1 (newest) ]
            // Indices: [0..keep | keep..idx           | idx..bufferSeqLen      ] <-- This is wrong, Python slices are different
            // Python slices:
            // keep part: v[..., :self.keep, :]
            // part 2 (newest inserted): v[..., self.keep : self._idx, :]
            // part 1 (older rotated): v[..., self._idx :, :]
            // Temporal Order: [ Keep | Part 1 | Part 2 ]

            var components = [MLXArray]()
            if keep > 0 {
                components.append(v[0..., 0..., ..<keep, 0...])  // Keep part
            }
            // Part 1 (older rotated part, from idx to end of buffer)
            if insertionIdx < bufferSeqLen {
                components.append(v[0..., 0..., insertionIdx ..< bufferSeqLen, 0...])
            }
            // Part 2 (newest inserted part, from keep up to idx)
            if keep < insertionIdx {
                components.append(v[0..., 0..., keep ..< insertionIdx, 0...])
            }

            if components.isEmpty {
                // Should not happen if v was not empty
                return MLXArray.zeros(like: v)[0..., 0..., ..<0, 0...]  // Return empty slice
            }

            let orderedV = MLX.concatenated(components, axis: 2)
            // Slice the reordered array to the logical offset
            let finalSliceOffset = min(effectiveOffset, orderedV.dim(2))
            return orderedV[0..., 0..., ..<finalSliceOffset, 0...]
        }
    }

    // Helper to trim the middle part (between keep and start of insertion)
    // Mimics Python's _trim function used in _update_concat
    private func trimMiddle(trimSize: Int, v: MLXArray) -> MLXArray {
        if trimSize <= 0 { return v }  // No trimming needed

        var components = [MLXArray]()

        // Part 1: Keep the first `keep` tokens
        if keep > 0 {
            // Ensure keep is within bounds
            let actualKeep = min(keep, v.dim(2))
            if actualKeep > 0 {
                components.append(v[0..., 0..., ..<actualKeep, 0...])
            }
        }

        // Part 2: Keep the elements *after* the trimmed section
        // Start index of the part to keep after trimming
        let remainingStart = keep + trimSize
        if remainingStart < v.dim(2) {
            components.append(v[0..., 0..., remainingStart..., 0...])
        }

        // Concatenate the kept parts
        if components.isEmpty {
            // Everything was trimmed, return an empty array slice
            return MLXArray.zeros(like: v)[0..., 0..., ..<0, 0...]
        } else if components.count == 1 {
            return components[0]  // Only one part remained
        } else {
            return MLX.concatenated(components, axis: 2)
        }
    }
}

// MARK: - Helper Functions

/// Creates an additive causal mask for attention.
///
/// Creates mask for `[B, H, N, N + Offset]` where `N` is `n`.
///
/// - Parameters:
///   - n: The sequence length of the query.
///   - offset: The offset for the key/value sequence length.
/// - Returns: An MLXArray suitable for adding to attention scores.
public func createAdditiveCausalMask(n: Int, offset: Int) -> MLXArray {
    let queryIndices = MLXArray(Int32(offset) ..< Int32(offset + n))  // Shape [N]
    let keyIndices = MLXArray(Int32(0) ..< Int32(offset + n))  // Shape [N + Offset]

    // Compare queryIndices[i] < keyIndices[j]
    // queryIndices shape: [N, 1]
    // keyIndices shape:   [1, N + Offset]
    // Result shape:       [N, N + Offset]
    let mask = queryIndices.expandedDimensions(axis: 1) .< keyIndices.expandedDimensions(axis: 0)

    // Add dimensions for batch and head: [1, 1, N, N + Offset]
    let mask4D = mask.expandedDimensions(axes: [0, 1])

    // Use a large negative number for masked positions
    // Multiplying by float directly handles type promotion if mask is bool
    return mask4D * Float(-1e9)
}

/// Creates an attention mask based on input sequence length and cache offset.
///
/// Only creates a mask for multi-token inputs (prefill phase). For single-token
/// generation (t=1), no explicit mask is typically needed as attention is only
/// computed for the single query token against all keys.
///
/// - Parameters:
///   - h: The input tensor to the attention layer. Expected shape [Batch, SeqLen, HiddenDim]
///        or [Batch, Heads, SeqLen, HeadDim]. `SeqLen` is extracted from `dim(1)` or `dim(2)`.
///   - cache: An array of KVCache instances (one per layer). Used to determine the offset.
///            Assumes all caches have the same offset.
///   - seqLenDim: The dimension index representing sequence length in `h`. Typically 1 or 2.
/// - Returns: An optional MLXArray attention mask, or nil if `t <= 1`.
public func createAttentionMask(h: MLXArray, cache: [any KVCache]?, seqLenDim: Int = 1) -> MLXArray?
{
    guard h.ndim > seqLenDim else {
        // Handle cases where input tensor doesn't have expected dimensions
        print(
            "Warning: Input tensor `h` has fewer dimensions than expected (\(h.ndim) vs \(seqLenDim + 1)). Cannot determine sequence length."
        )
        return nil
    }
    let t = h.dim(seqLenDim)  // Extract sequence length

    // Only create mask for multi-token inputs (prefill)
    if t > 1 {
        var offset = 0
        // Get offset from the first cache entry
        if let firstCache = cache?.first {
            offset = firstCache.offset
        }
        // Use the refined createAdditiveCausalMask which returns 4D mask
        return createAdditiveCausalMask(n: t, offset: offset).asType(h.dtype)
    }
    // No mask needed for single-token generation
    return nil
}

// MARK: - KVCacheSimple

public class KVCacheSimple: KVCache, Evaluatable {
    var keys: MLXArray?
    var values: MLXArray?

    public var offset = 0
    var step = 256  // Resizing step size

    public init() {}

    public func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset
        let newSeqLen = keys.dim(2)  // Assuming [B, H, L, D]

        // Check if resizing is needed
        let needsResize: Bool
        if let currentKeys = self.keys {
            needsResize = (previous + newSeqLen) > currentKeys.dim(2)  // Check if new length exceeds current capacity
        } else {
            needsResize = true  // Needs allocation if keys is nil
        }

        if needsResize {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            // Calculate new size based on steps
            let requiredLength = previous + newSeqLen
            let nSteps = (requiredLength + step - 1) / step  // Number of steps needed
            let newCapacity = nSteps * step

            let kShape = [B, kvHeads, newCapacity, kHeadDim]
            let vShape = [B, kvHeads, newCapacity, vHeadDim]

            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            // Copy existing data if it exists
            if let currentKeys = self.keys, let currentValues = self.values, previous > 0 {
                // Copy only the valid part of the old cache
                newK[0..., 0..., ..<previous, 0...] = currentKeys[0..., 0..., ..<previous, 0...]
                newV[0..., 0..., ..<previous, 0...] = currentValues[0..., 0..., ..<previous, 0...]
            }
            self.keys = newK
            self.values = newV
        }

        // Insert the new keys/values
        // Use optional chaining just in case allocation failed, though it shouldn't
        self.keys?[0..., 0..., previous ..< (previous + newSeqLen), 0...] = keys
        self.values?[0..., 0..., previous ..< (previous + newSeqLen), 0...] = values

        // Update offset
        self.offset += newSeqLen

        // Return the valid portion of the cache up to the new offset
        // Guard against nil arrays before slicing and returning
        guard let currentKeys = self.keys, let currentValues = self.values else {
            // This should ideally not happen if allocation succeeded or was not needed
            fatalError("Cache arrays are unexpectedly nil after update.")
        }

        return (
            currentKeys[0..., 0..., ..<self.offset, 0...],
            currentValues[0..., 0..., ..<self.offset, 0...]
        )
    }

    /// Checks if the cache can be logically trimmed. Always true for simple cache.
    public func isTrimmable() -> Bool {
        return true
    }

    /// Trims the cache state logically by reducing the offset.
    /// Does not shrink allocated memory.
    /// - Parameter count: The number of tokens to trim from the end.
    /// - Returns: The actual number of tokens trimmed.
    public func trim(count: Int) -> Int {
        let trimmedCount = min(self.offset, count)  // Don't trim more than available
        self.offset -= trimmedCount
        return trimmedCount
    }
}
