// Copyright © 2025 Apple Inc.

import Foundation
import ReerCodable

/// Base ``LanguageModel`` configuration -- provides `modelType`
/// and `quantization` (used in loading the model).
///
/// This is used by ``ModelFactory/load(hub:configuration:progressHandler:)``
/// to determine the type of model to load.
@Codable(memberwiseInit: false)
public struct BaseConfiguration: Sendable {
    @CodingKey("model_type") public let modelType: String

    @Codable
    public struct Quantization: Sendable, Equatable {
        @CodingKey("group_size") public let groupSize: Int
        public let bits: Int
        @CodingKey("quant_method") public var quantMethod: String? = nil
        @CodingKey("linear_class") public var linearClass: String? = nil
        @CodingKey("quantization_mode") public var quantizationMode: String? = nil

        public var asTuple: (Int, Int) { (groupSize, bits) }
    }

    /// handling instructions for ``PerLayerQuantization``
    public enum QuantizationOption: Sendable {
        case skip
        case quantize(Quantization)
    }

    /// Per-layer ``Quantization`` values with optional default.
    public struct PerLayerQuantization: Sendable {
        public var quantization: Quantization? = nil
        public var perLayerQuantization: [String: QuantizationOption]

        public init(
            quantization: BaseConfiguration.Quantization? = nil,
            perLayerQuantization: [String: BaseConfiguration.QuantizationOption]
        ) {
            self.quantization = quantization
            self.perLayerQuantization = perLayerQuantization
        }

        /// The quantization to apply for the given layer name or nil for no quantization.
        public func quantization(layer: String) -> Quantization? {
            if let perLayer = perLayerQuantization[layer] {
                switch perLayer {
                case .skip:
                    return nil
                case .quantize(let quantization):
                    return quantization
                }
            } else {
                return quantization
            }
        }
    }

    /// Special codable to support a mixed key: Int / key: Quantization
    /// structure for hereogenous quantization, e.g.
    ///
    /// ```
    /// "quantization": {
    ///     "group_size": 64,
    ///     "bits": 4,
    ///     "model.embed_tokens": {
    ///         "group_size": 32,
    ///         "bits": 4
    ///     },
    ///     "model.layers.0.self_attn.q_norm": false,
    /// ```
    ///
    /// This mixed type structure requires manual decoding.
    struct QuantizationContainer: Codable, Sendable {
        var quantization: Quantization?
        var perLayerQuantization: PerLayerQuantization

        internal init(quantization: Quantization?, perLayerQuantization: PerLayerQuantization) {
            self.quantization = quantization
            self.perLayerQuantization = perLayerQuantization
        }

        init(from decoder: any Decoder) throws {
            // handle the embedded Quantization
            self.quantization = try? Quantization(from: decoder)

            // and the interleaved per-layer values
            var perLayerQuantization = [String: QuantizationOption]()
            let container = try decoder.container(keyedBy: AnyCodingKey.self)
            for key in container.allKeys {
                switch key.stringValue {
                // ignore keys that belong to Quantization
                case "group_size", "bits": continue
                case "quant_method", "linear_class", "quantization_mode": continue

                default:
                    if let f = try? container.decode(Bool.self, forKey: key) {
                        if !f {
                            perLayerQuantization[key.stringValue] = .skip
                        }
                    } else {
                        perLayerQuantization[key.stringValue] = .quantize(
                            try container.decode(Quantization.self, forKey: key))
                    }
                }
            }
            self.perLayerQuantization = PerLayerQuantization(
                quantization: quantization, perLayerQuantization: perLayerQuantization)
        }

        func encode(to encoder: any Encoder) throws {
            try quantization.encode(to: encoder)

            var container = encoder.container(keyedBy: AnyCodingKey.self)
            for (key, value) in perLayerQuantization.perLayerQuantization {
                guard let key = AnyCodingKey(stringValue: key) else { continue }
                switch value {
                case .skip:
                    try container.encode(false, forKey: key)
                case .quantize(let q):
                    try container.encode(q, forKey: key)
                }
            }
        }
    }

    @CodingKey("quantization") var quantizationContainer: QuantizationContainer?

    @available(*, deprecated, message: "Please use perLayerQuantization instead")
    public var quantization: Quantization? {
        quantizationContainer?.quantization
    }

    public var perLayerQuantization: PerLayerQuantization? {
        quantizationContainer?.perLayerQuantization
    }

    public init(
        modelType: String, quantization: Quantization? = nil,
        perLayerQuantization: PerLayerQuantization? = nil
    ) {
        self.modelType = modelType
        self.quantizationContainer = QuantizationContainer(
            quantization: quantization,
            perLayerQuantization: perLayerQuantization ?? .init(perLayerQuantization: [:]))
    }
}
