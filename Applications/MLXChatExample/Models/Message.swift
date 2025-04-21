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

    /// The text content of the message
    var content: String

    /// Array of image URLs attached to the message
    var images: [URL]

    /// Array of video URLs attached to the message
    var videos: [URL]

    /// Timestamp when the message was created
    let timestamp: Date

    /// Creates a new message with the specified role, content, and optional media attachments
    /// - Parameters:
    ///   - role: The role of the message sender
    ///   - content: The text content of the message
    ///   - images: Optional array of image URLs
    ///   - videos: Optional array of video URLs
    init(role: Role, content: String, images: [URL] = [], videos: [URL] = []) {
        self.id = UUID()
        self.role = role
        self.content = content
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
