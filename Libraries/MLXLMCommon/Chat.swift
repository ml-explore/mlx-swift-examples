// Copyright © 2025 Apple Inc.

public enum Chat {
    public struct Message {
        /// The role of the message sender.
        public let role: Role

        /// The content of the message.
        public let content: String

        /// Array of image data associated with the message.
        public let images: [UserInput.Image]

        /// Array of video data associated with the message.
        public let videos: [UserInput.Video]

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

        public enum Role: String {
            case user
            case assistant
            case system
        }
    }
}

public protocol MessageGenerator {
    /// Returns [String: Any] aka Message
    func generate(message: Chat.Message) -> Message
}

extension MessageGenerator {
    /// Returns array of [String: Any] aka Message
    public func generate(messages: [Chat.Message]) -> [Message] {
        var rawMessages: [Message] = []

        for message in messages {
            let raw = generate(message: message)
            rawMessages.append(raw)
        }

        return rawMessages
    }
}

public struct DefaultMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        [
            "role": message.role.rawValue,
            "content": message.content,
        ]
    }
}

public struct Qwen2VLMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        [
            "role": message.role.rawValue,
            "content": [
                ["type": "text", "text": message.content]
            ]
                // Messages format for Qwen 2 VL, Qwen 2.5 VL. May need to be adapted for other models.
                + message.images.map { _ in
                    ["type": "image"]
                }
                + message.videos.map { _ in
                    ["type": "video"]
                },
        ]
    }
}
