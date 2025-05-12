//
//  ConversationView.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import SwiftUI

/// Displays the chat conversation as a scrollable list of messages.
struct ConversationView: View {
    /// Array of messages to display in the conversation
    let messages: [Message]

    @Binding var displayMode: MessageDisplayMode

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                ForEach(messages) { message in
                    MessageView(message, displayMode: displayMode)
                        .padding(.horizontal, 12)
                }
            }
        }
        .padding(.vertical, 8)
        .defaultScrollAnchor(.bottom, for: .sizeChanges)
    }
}

#Preview {
    // Display sample conversation in preview
    ConversationView(
        messages: SampleData.conversation,
        displayMode: .constant(.markdown)
    )
}
