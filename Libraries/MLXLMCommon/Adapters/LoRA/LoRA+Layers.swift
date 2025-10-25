// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Implementation of LoRA `Linear` replacement layer.
///
/// This layer implements the LoRA capabilities for `Linear` layers, specifically:
///
/// - converting `Linear` or `QuantizedLinear` layers to ``LoRALinear`` / ``QLoRALinear``
/// - converting ``LoRALinear`` back to `Linear` or `QuantizedLinear` (``LoRAConvertToLinear``)
/// - implementing the LoRA evaluation
///
/// ``QLoRALinear`` is the equivalent class for `QuantizedLinear`.
///
/// This is not typically used directly -- ``LoRATrain/convert(model:layers:)`` is used to
/// add the adapter layers to a given model.
///
/// ### See Also
/// - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
/// - [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
/// - ``QLoRALinear``
/// - ``LoRATrain/convert(model:layers:)``
/// - ``LoRATrain/fuse(model:layers:deQuantize:)``
public class LoRALinear: Linear, LoRALayer {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    required public init(
        _ inputDimensions: Int, _ outputDimensions: Int, rank: Int = 8, bias: Bool = false,
        scale: Float = 20.0, linear: Linear
    ) {
        // Scale for low-rank update
        self.scale = scale

        // Low rank lora weights
        let loraScale = 1 / sqrt(Float(inputDimensions))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

        super.init(weight: linear.weight, bias: linear.bias)

        freeze()
    }

    /// Freeze all parameters except the lora parameters
    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        // realize the keys and omit the lora parameters
        let keys =
            (keys ?? self.filterMap(filter: Self.filterLocalParameters).flattened().map { $0.0 })
            .filter {
                $0 != "lora_a" && $0 != "lora_b"
            }
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    /// Convert a `Linear` or `QuantizedLinear` layer into a new `Linear` layer
    /// that implements the `LoRA` adapter.
    ///
    /// This is typically called via ``LoRATrain/convert(model:layers:)``.
    ///
    /// ### See Also
    /// - ``LoRATrain/convert(model:layers:)``
    /// - ``QLoRALinear/from(linear:rank:)``
    public static func from(linear: Linear, rank: Int = 8, scale: Float = 20.0) -> LoRALayer {
        if let linear = linear as? QuantizedLinear {
            return QLoRALinear.from(linear: linear, rank: rank, scale: scale)
        }
        let (outputDimensions, inputDimensions) = linear.shape
        return LoRALinear(
            inputDimensions, outputDimensions, rank: rank, scale: scale, linear: linear)
    }

    /// Convert back into a fused `Linear` layer.
    ///
    /// This is typically called via ``LoRATrain/fuse(model:layers:deQuantize:)``.
    ///
    /// ### See Also
    /// - ``LoRATrain/fuse(model:layers:deQuantize:)``
    /// - ``LoRAConvertToLinear``
    /// - ``QLoRALinear/toLinear(deQuantize:)``
    public func fused() -> Module {
        let dtype = weight.dtype
        let loraB = (scale * loraB.T).asType(dtype)
        let loraA = loraA.T.asType(dtype)
        return Linear(weight: weight + matmul(loraB, loraA), bias: bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x.asType(weight.dtype))
        let z = matmul(matmul(x, self.loraA), self.loraB)
        return y + scale * z
    }
}

/// Implementation of LoRA `QuantizedLinear` replacement layer.
///
/// See ``LoRALinear`` (equivalent class for `Linear` layers) for more information.
public class QLoRALinear: QuantizedLinear, LoRALayer {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    required public init(
        _ inputDimensions: Int, _ outputDimensions: Int, rank: Int = 8, bias: Bool = false,
        scale: Float = 20.0, linear: QuantizedLinear
    ) {

        // Scale for low-rank update
        self.scale = scale

        // Low rank lora weights
        let loraScale = 1 / sqrt(Float(inputDimensions))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

        super.init(
            weight: linear.weight, bias: linear.bias, scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits)

        // start frozen except for the lora keys
        freeze()
    }

    /// Freeze all parameters except the lora parameters
    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        // realize the keys and omit the lora parameters
        let keys =
            (keys ?? self.filterMap(filter: Self.filterLocalParameters).flattened().map { $0.0 })
            .filter {
                $0 != "lora_a" && $0 != "lora_b"
            }
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    /// Convert a `QuantizedLinear` layer into a new `Linear` layer
    /// that implements the `LoRA` adapter.
    ///
    /// This is typically called via ``LoRATrain/convert(model:layers:)``.
    ///
    /// ### See Also
    /// - ``LoRATrain/convert(model:layers:)``
    /// - ``LoRALinear/from(linear:rank:)``
    public static func from(linear: QuantizedLinear, rank: Int = 8, scale: Float = 20.0)
        -> LoRALayer
    {
        let (outputDimensions, inputDimensions) = linear.shape
        return QLoRALinear(
            inputDimensions, outputDimensions, rank: rank, scale: scale, linear: linear)
    }

    /// Convert back into a fused `QuantizedLinear` layer.
    ///
    /// This is typically called via ``LoRATrain/fuse(model:layers:deQuantize:)``.
    ///
    /// ### See Also
    /// - ``LoRATrain/fuse(model:layers:deQuantize:)``
    public func fused() -> Module {
        let weight = dequantizedWeight
        let dtype = dequantizedWeight.dtype
        let loraB = (scale * loraB.T).asType(dtype)
        let loraA = loraA.T.asType(dtype)
        return QuantizedLinear(
            weight: weight + matmul(loraB, loraA),
            bias: bias,
            groupSize: groupSize,
            bits: bits
        )
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x.asType(scales.dtype))
        let z = matmul(matmul(x, self.loraA), self.loraB)
        return y + scale * z
    }
}
