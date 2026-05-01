//
//  MessageSegment.swift
//  MLXChatExample
//

import Foundation

/// One on-screen region inside an assistant ``Message``: a run of content
/// text, a run of reasoning text, or a tool call. Stored in arrival order
/// so the UI renders reasoning, tool-call cards, and answer text
/// interleaved as the parser surfaced them.
enum MessageSegment: Identifiable {
    case content(TextSegment)
    case reasoning(TextSegment)
    case toolCall(ToolCall)

    var id: String {
        switch self {
        case .content(let segment), .reasoning(let segment):
            return segment.id.uuidString
        case .toolCall(let toolCall):
            return toolCall.id
        }
    }
}

/// A growable text run inside a ``MessageSegment``. `itemId` is the
/// parser's `msg_…` / `rs_…` id, used to merge consecutive deltas of
/// the same item into one segment. `id` is a stable UUID for SwiftUI
/// `Identifiable` – distinct from `itemId` to stay safe if the parser
/// ever reuses an id across non-contiguous spans.
@Observable
class TextSegment: Identifiable {
    let id = UUID()
    let itemId: String
    var text: String

    init(itemId: String, text: String = "") {
        self.itemId = itemId
        self.text = text
    }
}
