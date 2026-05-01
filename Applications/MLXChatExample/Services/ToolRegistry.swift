//
//  ToolRegistry.swift
//  MLXChatExample
//

import Foundation
import LMResponseParserMLX

/// Tool palette advertised to `ResponseChatSession`. Both tools are
/// side-effect-free: `get_current_datetime` returns the formatted current
/// time, and `save_file` is a no-op stub whose long `content` parameter
/// gives the streaming-arguments view something substantial to render
/// token-by-token. Replace the `save_file` body with real I/O if you
/// want persistence; ``dispatch(_:)`` is the natural place to add
/// argument validation and policy checks.
@MainActor
struct ToolRegistry {
    static let getCurrentDatetime: ToolSpec = [
        "type": "function",
        "function": [
            "name": "get_current_datetime",
            "description": "Get the current date and time as a formatted string.",
            "parameters": [
                "type": "object",
                "properties": [:] as [String: any Sendable],
                "required": [] as [String],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    static let saveFile: ToolSpec = [
        "type": "function",
        "function": [
            "name": "save_file",
            "description":
                "Save a file with the given name and content. Returns a confirmation string.",
            "parameters": [
                "type": "object",
                "properties": [
                    "name": [
                        "type": "string",
                        "description": "File name to save under.",
                    ] as [String: any Sendable],
                    "content": [
                        "type": "string",
                        "description": "Full text content of the file.",
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
                "required": ["name", "content"] as [String],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    static let allSpecs: [ToolSpec] = [getCurrentDatetime, saveFile]

    /// Dispatch a model-issued call. Unknown names and decode failures
    /// return model-visible error strings so the next pass can correct,
    /// rather than aborting the response.
    static func dispatch(_ call: ResponseFunctionToolCall) async -> String {
        switch call.name {
        case "get_current_datetime":
            return Date.now.formatted()

        case "save_file":
            struct Args: Decodable {
                let name: String
                let content: String
            }
            do {
                let args = try call.decodedArguments(as: Args.self)
                return "Saved \(args.name) (\(args.content.count) characters)"
            } catch {
                return "Error: could not decode save_file arguments: \(error.localizedDescription)"
            }

        default:
            return "Error: unknown tool '\(call.name)'."
        }
    }
}
