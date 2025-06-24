// Copyright © 2025 Apple Inc.

public enum Chat {
    public struct Message {
        /// The role of the message sender.
        public var role: Role

        /// The content of the message.
        public var content: String

        /// Array of image data associated with the message.
        public var images: [UserInput.Image]

        /// Array of video data associated with the message.
        public var videos: [UserInput.Video]

        public init(
            role: Role, content: String, images: [UserInput.Image] = [],
            videos: [UserInput.Video] = []
        ) {
            self.role = role
            self.content = content
            self.images = images
            self.videos = videos
        }

        public static func system(
            _ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = []
        ) -> Self {
            Self(role: .system, content: content, images: images, videos: videos)
        }

        public static func assistant(
            _ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = []
        ) -> Self {
            Self(role: .assistant, content: content, images: images, videos: videos)
        }

        public static func user(
            _ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = []
        ) -> Self {
            Self(role: .user, content: content, images: images, videos: videos)
        }

        public static func tool(_ content: String) -> Self {
            Self(role: .tool, content: content)
        }

        public enum Role: String {
            case user
            case assistant
            case system
            case tool
        }
    }
}

/// Protocol for something that can convert structured
/// ``Chat.Message`` into model specific ``Message``
/// (raw dictionary) format.
///
/// Typically this is owned and used by a ``UserInputProcessor``:
///
/// ```swift
/// public func prepare(input: UserInput) async throws -> LMInput {
///     let messages = Qwen2VLMessageGenerator().generate(from: input)
///     ...
/// ```
public protocol MessageGenerator {

    /// Generates messages from the input.
    func generate(from input: UserInput) -> [Message]

    /// Returns array of `[String: Any]` aka ``Message``
    func generate(messages: [Chat.Message]) -> [Message]

    /// Returns `[String: Any]` aka ``Message``.
    func generate(message: Chat.Message) -> Message
}

extension MessageGenerator {

    public func generate(message: Chat.Message) -> Message {
        [
            "role": message.role.rawValue,
            "content": message.content,
        ]
    }

    public func generate(messages: [Chat.Message]) -> [Message] {
        var rawMessages: [Message] = []

        for message in messages {
            let raw = generate(message: message)
            rawMessages.append(raw)
        }

        return rawMessages
    }

    public func generate(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .text(let text):
            generate(messages: [.user(text)])
        case .messages(let messages):
            messages
        case .chat(let messages):
            generate(messages: messages)
        }
    }
}

/// Default implementation of ``MessageGenerator`` that produces a
/// `role` and `content`.
///
/// ```swift
/// [
///     "role": message.role.rawValue,
///     "content": message.content,
/// ]
/// ```
public struct DefaultMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        [
            "role": message.role.rawValue,
            "content": message.content,
        ]
    }
}

/// Implementation of ``MessageGenerator`` that produces a
/// `role` and `content` but omits `system` roles.
///
/// ```swift
/// [
///     "role": message.role.rawValue,
///     "content": message.content,
/// ]
/// ```
public struct NoSystemMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(messages: [Chat.Message]) -> [Message] {
        messages
            .filter { $0.role != .system }
            .map { generate(message: $0) }
    }
}
