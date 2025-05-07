//
//  MessageView.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import AVKit
import MarkdownUI
import SwiftUI

/// A view that displays a single message in the chat interface.
/// Supports different message roles (user, assistant, system) and media attachments.
struct MessageView: View {
    /// The message to be displayed
    let message: Message

    var displayMode: MessageDisplayMode = .markdown

    /// Creates a message view
    /// - Parameter message: The message model to display
    init(_ message: Message, displayMode: MessageDisplayMode = .markdown) {
        self.message = message
        self.displayMode = displayMode
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

                    // Message content with tinted background
                    Text(message.content)
                        .padding(.vertical, 8)
                        .padding(.horizontal, 12)
                        .background(.tint, in: .rect(cornerRadius: 16))
                        .textSelection(.enabled)
                }
            }

        case .assistant:
            // Assistant messages are left-aligned without background
            HStack {
                if displayMode == .markdown {
                    Markdown(message.content)
                        .textSelection(.enabled)
                } else {
                    Text(message.content)
                        .textSelection(.enabled)
                }

                Spacer()
            }

        case .system:
            // System messages are centered with computer icon
            Label(message.content, systemImage: "desktopcomputer")
                .font(.headline)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .center)
        }
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
