//
//  Message.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import Foundation

/// Represents a chat message in the conversation.
/// Messages can contain text content and optional media attachments (images and videos).
@Observable
class Message: Identifiable {
    /// Unique identifier for the message
    let id: UUID

    /// The role of the message sender (user, assistant, or system)
    let role: Role

    /// On-screen regions in arrival order. User/system messages have a
    /// single ``MessageSegment/content(_:)`` holding the whole text.
    /// Assistant messages grow as the parser surfaces reasoning runs,
    /// tool-call items, and content runs, interleaved.
    var segments: [MessageSegment]

    /// Array of image URLs attached to the message
    var images: [URL]

    /// Array of video URLs attached to the message
    var videos: [URL]

    /// Timestamp when the message was created
    let timestamp: Date

    /// Concatenated text from every ``MessageSegment/content(_:)`` segment.
    /// Used for user/system rendering and for re-seeding assistant turns
    /// into a `Chat.Message` history – the model only needs to see what
    /// it actually said, not its reasoning or tool-call wire syntax.
    var content: String {
        segments.compactMap {
            if case let .content(segment) = $0 { return segment.text }
            return nil
        }.joined()
    }

    /// Creates a new message with the specified role, content, and optional media attachments
    /// - Parameters:
    ///   - role: The role of the message sender
    ///   - content: The text content of the message
    ///   - images: Optional array of image URLs
    ///   - videos: Optional array of video URLs
    init(role: Role, content: String, images: [URL] = [], videos: [URL] = []) {
        self.id = UUID()
        self.role = role
        self.segments = content.isEmpty
            ? []
            : [.content(TextSegment(itemId: "_initial", text: content))]
        self.images = images
        self.videos = videos
        self.timestamp = .now
    }

    /// Defines the role of the message sender in the conversation
    enum Role {
        /// Message from the user
        case user
        /// Message from the AI assistant
        case assistant
        /// System message providing context or instructions
        case system
    }
}

/// Convenience methods for creating different types of messages
extension Message {
    /// Creates a user message with optional media attachments
    /// - Parameters:
    ///   - content: The text content of the message
    ///   - images: Optional array of image URLs
    ///   - videos: Optional array of video URLs
    /// - Returns: A new Message instance with user role
    static func user(_ content: String, images: [URL] = [], videos: [URL] = []) -> Message {
        Message(role: .user, content: content, images: images, videos: videos)
    }

    /// Creates an assistant message
    /// - Parameter content: The text content of the message
    /// - Returns: A new Message instance with assistant role
    static func assistant(_ content: String) -> Message {
        Message(role: .assistant, content: content)
    }

    /// Creates a system message
    /// - Parameter content: The text content of the message
    /// - Returns: A new Message instance with system role
    static func system(_ content: String) -> Message {
        Message(role: .system, content: content)
    }
}
