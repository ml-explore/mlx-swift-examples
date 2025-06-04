// Copyright © 2025 Apple Inc.

import Foundation

/// Type-safe representation of JSON values
public enum JSONValue: Hashable, Codable, Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if container.decodeNil() {
            self = .null
        } else if let bool = try? container.decode(Bool.self) {
            self = .bool(bool)
        } else if let int = try? container.decode(Int.self) {
            self = .int(int)
        } else if let double = try? container.decode(Double.self) {
            self = .double(double)
        } else if let string = try? container.decode(String.self) {
            self = .string(string)
        } else if let array = try? container.decode([JSONValue].self) {
            self = .array(array)
        } else if let object = try? container.decode([String: JSONValue].self) {
            self = .object(object)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container, debugDescription: "Cannot decode JSON value")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()

        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let value):
            try container.encode(value)
        case .int(let value):
            try container.encode(value)
        case .double(let value):
            try container.encode(value)
        case .string(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        }
    }

    public static func from(_ value: Any) -> JSONValue {
        switch value {
        case is NSNull:
            return .null
        case let bool as Bool:
            return .bool(bool)
        case let int as Int:
            return .int(int)
        case let double as Double:
            return .double(double)
        case let string as String:
            return .string(string)
        case let array as [Any]:
            return .array(array.map { from($0) })
        case let dict as [String: Any]:
            var result = [String: JSONValue]()
            for (key, value) in dict {
                result[key] = from(value)
            }
            return .object(result)
        default:
            return .string(String(describing: value))
        }
    }

    public var anyValue: Any {
        switch self {
        case .null:
            return NSNull()
        case .bool(let value):
            return value
        case .int(let value):
            return value
        case .double(let value):
            return value
        case .string(let value):
            return value
        case .array(let value):
            return value.map { $0.anyValue }
        case .object(let value):
            return value.mapValues { $0.anyValue }
        }
    }

    /// Convert to JSON Schema representation
    public var asSchema: [String: Any] {
        switch self {
        case .null:
            return ["type": "null"]
        case .bool:
            return ["type": "boolean"]
        case .int:
            return ["type": "integer"]
        case .double:
            return ["type": "number"]
        case .string:
            return ["type": "string"]
        case .array(let elements):
            if let first = elements.first {
                return ["type": "array", "items": first.asSchema]
            }
            return ["type": "array"]
        case .object(let properties):
            var props: [String: Any] = [:]

            for (key, value) in properties {
                props[key] = value.asSchema
            }

            return ["type": "object", "properties": props]
        }
    }
}
