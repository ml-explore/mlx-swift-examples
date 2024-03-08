// Copyright © 2024 Apple Inc.

import Foundation

public enum StringOrNumber: Codable, Equatable {
    case string(String)
    case float(Float)

    public init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer()

        if let v = try? values.decode(Float.self) {
            self = .float(v)
        } else {
            let v = try values.decode(String.self)
            self = .string(v)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .float(let v): try container.encode(v)
        }
    }
}

public enum ModelType: String, Codable {
    case mistral
    case llama
    case phi
    case gemma
    case qwen2
    case starcoder2

    func createModel(configuration: URL) throws -> LLMModel {
        switch self {
        case .mistral, .llama:
            let configuration = try JSONDecoder().decode(
                LlamaConfiguration.self, from: Data(contentsOf: configuration))
            return LlamaModel(configuration)
        case .phi:
            let configuration = try JSONDecoder().decode(
                PhiConfiguration.self, from: Data(contentsOf: configuration))
            return PhiModel(configuration)
        case .gemma:
            let configuration = try JSONDecoder().decode(
                GemmaConfiguration.self, from: Data(contentsOf: configuration))
            return GemmaModel(configuration)
        case .qwen2:
            let configuration = try JSONDecoder().decode(
                Qwen2Configuration.self, from: Data(contentsOf: configuration))
            return Qwen2Model(configuration)
        case .starcoder2:
            let configuration = try JSONDecoder().decode(
                Starcoder2Configuration.self, from: Data(contentsOf: configuration))
            return Starcoder2Model(configuration)
        }
    }
}

public struct BaseConfiguration: Codable {
    let modelType: ModelType

    public struct Quantization: Codable {
        public init(groupSize: Int, bits: Int) {
            self.groupSize = groupSize
            self.bits = bits
        }

        let groupSize: Int
        let bits: Int

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits = "bits"
        }
    }

    var quantization: Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantization
    }
}
