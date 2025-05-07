// Copyright © 2025 Apple Inc.

import Foundation

public typealias ToolSpec = [String: Any] /* ? Should we change it to `any (Codable & Sendable)` */

public indirect enum ToolParameterType {
    case null
    case bool
    case int
    case double
    case string
    case data
    case array(ToolParameterType)
    case object([String: ToolParameterType])
}

public struct ToolParameter {
    public let name: String
    public let type: ToolParameterType
    public let description: String
    public let isRequired: Bool
    public let extraProperties: [String: Any]

    public static func required(_ name: String, type: ToolParameterType, description: String, extraProperties: [String: Any] = [:]) -> ToolParameter {
        ToolParameter(name: name, type: type, description: description, isRequired: true, extraProperties: extraProperties)
    }

    public static func optional(_ name: String, type: ToolParameterType, description: String, extraProperties: [String: Any] = [:]) -> ToolParameter {
        ToolParameter(name: name, type: type, description: description, isRequired: false, extraProperties: extraProperties)
    }
}

/// Protocol defining the requirements for a tool.
public protocol ToolProtocol: Sendable {
    /// The JSON Schema describing the tool's interface.
    var schema: ToolSpec { get }
}

public struct Tool<Input: Codable, Output: Codable>: ToolProtocol {
    public let schema: ToolSpec
    public let handler: (Input) async throws -> Output

    public init(
        name: String,
        description: String,
        parameters: [ToolParameter],
        handler: @escaping (Input) async throws -> Output
    ) {
        self.schema = Self.toFunctionCallSchema(name: name, description: description, parameters: parameters)
        self.handler = handler
    }

    public init(
        schema: ToolSpec,
        handler: @escaping (Input) async throws -> Output
    ) {
        self.schema = schema
        self.handler = handler
    }
}

extension Tool {
    private static func schema(from type: ToolParameterType) -> ToolSpec {
        switch type {
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
        case .data:
            return ["type": "string", "contentEncoding": "base64"]
        case .array(let element):
            return [
                "type": "array",
                "items": schema(from: element)
            ]
        case .object(let properties):
            var props: [String: Any] = [:]
            var requiredKeys: [String] = []

            for (key, value) in properties {
                props[key] = schema(from: value)
                requiredKeys.append(key)
            }

            return [
                "type": "object",
                "properties": props,
                "required": requiredKeys
            ]
        }
    }

    private static func toFunctionCallSchema(name: String, description: String, parameters: [ToolParameter]) -> ToolSpec {
        var properties: [String: Any] = [:]
        var requiredParams: [String] = []

        for param in parameters {
            var paramSchema = schema(from: param.type)

            // Add description to the schema
            paramSchema["description"] = param.description

            // Merge any extra JSON Schema properties, such as "enum", "default", "examples", etc.
            for (key, value) in param.extraProperties {
                paramSchema[key] = value
            }

            // Add parameter schema
            properties[param.name] = paramSchema

            if param.isRequired {
                requiredParams.append(param.name)
            }
        }

        let schema: ToolSpec = [
            "type": "function",
            "function": [
                "name": name,
                "description": description,
                "parameters": [
                    "type": "object",
                    "properties": properties,
                    "required": requiredParams
                ]
            ]
        ]

        return schema
    }
}
