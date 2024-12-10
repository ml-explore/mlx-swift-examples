// Copyright © 2024 Apple Inc.

import CoreImage
import Foundation
import MLX

/// Container for raw user input.
///
/// A ``UserInputProcessor`` can convert this to ``LMInput``.
/// See also ``ModelContext``.
public struct UserInput: Sendable {

    /// Representation of a prompt or series of messages (conversation).
    public enum Prompt: Sendable, CustomStringConvertible {
        case text(String)
        case messages([[String: String]])

        public func asMessages() -> [[String: String]] {
            switch self {
            case .text(let text):
                return [["role": "user", "content": text]]
            case .messages(let messages):
                return messages
            }
        }

        public var description: String {
            switch self {
            case .text(let text):
                return text
            case .messages(let messages):
                return messages.map { $0.description }.joined(separator: "\n")
            }
        }
    }

    /// Representation of a single image.
    public enum Image: Sendable {
        case ciImage(CIImage)
        case url(URL)
        case array(MLXArray)

        public func asCIImage() throws -> CIImage {
            switch self {
            case .ciImage(let image):
                return image

            case .url(let url):
                if let image = CIImage(contentsOf: url) {
                    return image
                }
                throw UserInputError.unableToLoad(url)

            case .array(let array):
                guard array.ndim == 3 else {
                    throw UserInputError.arrayError("array must have 3 dimensions: \(array.ndim)")
                }

                var array = array

                // convert to 0 .. 255
                if array.max().item(Float.self) <= 1.0 {
                    array = array * 255
                }

                // planar -> pixels
                switch array.dim(0) {
                case 3, 4:
                    // channels first (planar)
                    array = array.transposed(1, 2, 0)
                default:
                    break
                }

                // 4 components per pixel
                switch array.dim(-1) {
                case 3:
                    // pad to 4 bytes per pixel
                    array = padded(array, widths: [0, 0, [0, 1]], value: MLXArray(255))
                case 4:
                    // good
                    break
                default:
                    throw UserInputError.arrayError(
                        "channel dimension must be last and 3/4: \(array.shape)")
                    break
                }

                let arrayData = array.asData()
                let (H, W, C) = array.shape3
                let cs = CGColorSpace(name: CGColorSpace.sRGB)!

                return CIImage(
                    bitmapData: arrayData.data, bytesPerRow: W * 4,
                    size: .init(width: W, height: H),
                    format: .RGBA8, colorSpace: cs)
            }
        }
    }

    /// Representation of processing to apply to media.
    public struct Processing: Sendable {
        public var resize: CGSize?

        public init(resize: CGSize? = nil) {
            self.resize = resize
        }
    }

    public var prompt: Prompt
    public var images = [Image]()
    public var processing: Processing = .init()

    public init(prompt: String, images: [Image] = [Image]()) {
        self.prompt = .text(prompt)
        self.images = images
    }

    public init(messages: [[String: String]], images: [Image] = [Image]()) {
        self.prompt = .messages(messages)
        self.images = images
    }

    public init(prompt: Prompt, images: [Image] = [Image](), processing: Processing = .init()) {
        self.prompt = prompt
        self.images = images
        self.processing = processing
    }
}

/// Protocol for a type that can convert ``UserInput`` to ``LMInput``.
///
/// See also ``ModelContext``.
public protocol UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput
}

private enum UserInputError: Error {
    case notImplemented
    case unableToLoad(URL)
    case arrayError(String)
}

/// A do-nothing ``UserInputProcessor``.
public struct StandInUserInputProcessor: UserInputProcessor {
    public init() {}

    public func prepare(input: UserInput) throws -> LMInput {
        throw UserInputError.notImplemented
    }
}
