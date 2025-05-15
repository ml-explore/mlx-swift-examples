// Copyright © 2025 Apple Inc.

public indirect enum ToolParameterType {
    case string
    case bool
    case int
    case double
    case array(elementType: ToolParameterType)
    case object(properties: [ToolParameter])
    case data

    var schemaType: [String: Any] {
        switch self {
        case .string: return ["type": "string"]
        case .bool: return ["type": "boolean"]
        case .int: return ["type": "integer"]
        case .double: return ["type": "number"]
        case .data: return ["type": "string", "contentEncoding": "base64"]
        case .array(let elementType):
            return ["type": "array", "items": elementType.schemaType]
        case .object(let properties):
            var props = [String: Any]()
            var required = [String]()

            for param in properties {
                props[param.name] = param.schema
                if param.isRequired {
                    required.append(param.name)
                }
            }

            return ["type": "object", "properties": props, "required": required]
        }
    }
}

public struct ToolParameter {
    public let name: String
    public let type: ToolParameterType
    public let description: String
    public let isRequired: Bool
    public let extraProperties: [String: Any]

    public var schema: [String: Any] {
        var schema = type.schemaType
        schema["description"] = description

        // Add extra properties
        for (key, value) in extraProperties {
            schema[key] = value
        }

        return schema
    }

    public static func required(
        _ name: String,
        type: ToolParameterType,
        description: String,
        extraProperties: [String: Any] = [:]
    ) -> ToolParameter {
        ToolParameter(
            name: name,
            type: type,
            description: description,
            isRequired: true,
            extraProperties: extraProperties
        )
    }

    public static func optional(
        _ name: String,
        type: ToolParameterType,
        description: String,
        extraProperties: [String: Any] = [:]
    ) -> ToolParameter {
        ToolParameter(
            name: name,
            type: type,
            description: description,
            isRequired: false,
            extraProperties: extraProperties
        )
    }
}
