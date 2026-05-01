//
//  MessageView.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import AVKit
import SwiftUI

/// A view that displays a single message in the chat interface.
/// Supports different message roles (user, assistant, system) and media attachments.
struct MessageView: View {
    /// The message to be displayed
    let message: Message

    /// Creates a message view
    /// - Parameter message: The message model to display
    init(_ message: Message) {
        self.message = message
    }

    var body: some View {
        switch message.role {
        case .user:
            // User messages are right-aligned with blue background
            HStack {
                Spacer()
                VStack(alignment: .trailing, spacing: 8) {
                    // Display first image if present
                    if let firstImage = message.images.first {
                        AsyncImage(url: firstImage) { image in
                            image
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                        } placeholder: {
                            ProgressView()
                        }
                        .frame(maxWidth: 250, maxHeight: 200)
                        .clipShape(.rect(cornerRadius: 12))
                    }

                    // Display first video if present
                    if let firstVideo = message.videos.first {
                        VideoPlayer(player: AVPlayer(url: firstVideo))
                            .frame(width: 250, height: 340)
                            .clipShape(.rect(cornerRadius: 12))
                    }

                    // Message content with tinted background.
                    // LocalizedStringKey used to trigger default handling of markdown content.
                    Text(LocalizedStringKey(message.content))
                        .padding(.vertical, 8)
                        .padding(.horizontal, 12)
                        .background(.tint, in: .rect(cornerRadius: 16))
                        .textSelection(.enabled)
                }
            }

        case .assistant:
            // Segments are rendered in arrival order so reasoning runs,
            // tool calls, and content interleave exactly as the parser
            // surfaced them. Empty-after-trim text segments are skipped
            // so VStack spacing doesn't reserve a row for them.
            HStack {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(message.segments) { segment in
                        switch segment {
                        case .reasoning(let textSegment):
                            let trimmed = textSegment.text.trimmingCharacters(
                                in: .whitespacesAndNewlines)
                            if !trimmed.isEmpty {
                                Text(trimmed)
                                    .font(.callout.italic())
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }
                        case .toolCall(let toolCall):
                            ToolCallCard(toolCall: toolCall)
                        case .content(let textSegment):
                            let trimmed = textSegment.text.trimmingCharacters(
                                in: .whitespacesAndNewlines)
                            if !trimmed.isEmpty {
                                // LocalizedStringKey triggers markdown rendering.
                                Text(LocalizedStringKey(trimmed))
                                    .textSelection(.enabled)
                            }
                        }
                    }
                }

                Spacer()
            }

        case .system:
            // System messages are centered, prefixed for clarity
            Text("\(Text("System prompt: ").font(.headline))\(message.content)")
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .center)
        }
    }
}

/// Card showing one tool call's name, arguments, and result. Arguments
/// pretty-print once the streaming JSON parses; until then the raw
/// partial buffer is shown so the buildup is visible.
private struct ToolCallCard: View {
    let toolCall: ToolCall

    /// `nil` when the call has no arguments (don't render the row).
    private var formattedArguments: String? {
        let raw = toolCall.argumentsRaw.trimmingCharacters(in: .whitespacesAndNewlines)
        if raw.isEmpty || raw == "{}" || raw == "[]" { return nil }
        guard let data = raw.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let pretty = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys]
              ),
              let string = String(data: pretty, encoding: .utf8)
        else {
            return raw
        }
        return string
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text("Tool call:")
                    .font(.headline)
                Text(toolCall.name)
                    .font(.subheadline.bold())
            }

            if let formattedArguments {
                Text(formattedArguments)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if let result = toolCall.result {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("Result:")
                        .font(.headline)
                    Text(result)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
        }
        .padding(10)
        .glassEffect(.regular, in: .rect(cornerRadius: 10))
    }
}

#Preview {
    VStack(spacing: 20) {
        MessageView(.system("You are a helpful assistant."))

        MessageView(
            .user(
                "Here's a photo",
                images: [URL(string: "https://picsum.photos/200")!]
            )
        )

        MessageView(.assistant("I see your photo!"))
    }
    .padding()
}
