// Copyright © 2025 Apple Inc.

import Foundation

/// Used to process generated text to detect tool calls and manage the generation flow
public class ToolCallProcessor {

    public init() {}

    // Track the current state of processing
    private enum State {
        case normal
        case potentialToolCall
        case collectingToolCall
    }

    private var state = State.normal
    private var toolCallBuffer = ""

    // Tags to detect
    private static let toolUseStartTag = "<tool_call>"
    private static let toolUseEndTag = "</tool_call>"

    private static let toolCallRegex = #/<tool_call>\s*(\{.*?\})\s*<\/tool_call>/#

    /// The current parsed tool call, if any
    public var toolCalls: [ToolCall] = []

    /// Append a generated text chunk and process for tool call tags
    /// - Parameter chunk: The text chunk to process
    /// - Returns: Any regular text that should be yielded (non-tool call content)
    public func processChunk(_ chunk: String) -> String? {
        guard (state == .normal && chunk.contains("<")) || state != .normal else {
            return chunk
        }

        toolCallBuffer += chunk
        var leadingToken: String?

        switch state {
        case .normal:
            // Change state to potential tool call
            state = .potentialToolCall

            leadingToken = separateToken(from: &toolCallBuffer, separator: "<", returnLeading: true)

            fallthrough
        case .potentialToolCall:
            if partialMatch(buffer: toolCallBuffer, tag: Self.toolUseStartTag) {
                if toolCallBuffer.starts(with: Self.toolUseStartTag) {
                    state = .collectingToolCall
                    fallthrough
                } else {
                    return nil
                }
            } else {
                // Otherwise, return the collected text and reset the state
                state = .normal
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                return (leadingToken ?? "") + buffer
            }
        case .collectingToolCall:
            if toolCallBuffer.contains(Self.toolUseEndTag) {
                // Separate the trailing token
                let trailingToken = separateToken(
                    from: &toolCallBuffer, separator: Self.toolUseEndTag, returnLeading: false)

                // Parse the tool call
                if let toolCall = parseToolCall(toolCallBuffer) {
                    toolCalls.append(toolCall)
                }

                state = .normal
                toolCallBuffer = ""

                // If the token contains a "<", there may be more tool calls to come
                if let trailingToken, trailingToken.contains("<") {
                    return processChunk(trailingToken)
                } else {
                    // Otherwise, return the collected token, or nil if it's empty
                    return trailingToken?.isEmpty ?? true ? nil : trailingToken
                }
            } else {
                return nil
            }
        }
    }

    /// Separates a token from a string buffer based on a separator
    /// - Parameters:
    ///   - buffer: The string buffer to modify
    ///   - separator: The separator string to search for
    ///   - returnLeading: If true, returns text before separator; if false, returns text after
    /// - Returns: The separated token, or nil if separator not found
    private func separateToken(from buffer: inout String, separator: String, returnLeading: Bool)
        -> String?
    {
        guard let range = buffer.range(of: separator) else { return nil }

        let token: String
        if returnLeading {
            token = String(buffer[..<range.lowerBound])
            buffer = String(buffer[range.lowerBound...])
        } else {
            token = String(buffer[range.upperBound...])
            buffer = String(buffer[..<range.upperBound])
        }

        return token
    }

    private func partialMatch(buffer: String, tag: String) -> Bool {
        for (tagIndex, bufferIndex) in zip(tag.indices, buffer.indices) {
            if buffer[bufferIndex] != tag[tagIndex] {
                return false
            }
        }

        return true
    }

    /// Parse a tool call from the content inside <tool_use> tags
    private func parseToolCall(_ content: String) -> ToolCall? {
        guard let match = content.firstMatch(of: Self.toolCallRegex) else { return nil }

        let jsonData = String(match.output.1).data(using: .utf8)!

        if let json = try? JSONDecoder().decode(ToolCall.Function.self, from: jsonData) {
            return ToolCall(function: json)
        } else {
            return nil
        }
    }
}
