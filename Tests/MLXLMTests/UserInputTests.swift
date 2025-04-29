import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

func assertEqual(
    _ v1: Any, _ v2: Any, path: [String] = [], file: StaticString = #filePath, line: UInt = #line
) {
    switch (v1, v2) {
    case let (v1, v2) as (String, String):
        XCTAssertEqual(v1, v2, file: file, line: line)

    case let (v1, v2) as ([Any], [Any]):
        XCTAssertEqual(
            v1.count, v2.count, "Arrays not equal size at \(path)", file: file, line: line)

        for (index, (v1v, v2v)) in zip(v1, v2).enumerated() {
            assertEqual(v1v, v2v, path: path + [index.description], file: file, line: line)
        }

    case let (v1, v2) as ([String: Any], [String: Any]):
        XCTAssertEqual(
            v1.keys.sorted(), v2.keys.sorted(),
            "\(String(describing: v1.keys.sorted())) and \(String(describing: v2.keys.sorted())) not equal at \(path)",
            file: file, line: line)

        for (k, v1v) in v1 {
            if let v2v = v2[k] {
                assertEqual(v1v, v2v, path: path + [k], file: file, line: line)
            } else {
                XCTFail("Missing value for \(k) at \(path)", file: file, line: line)
            }
        }
    default:
        XCTFail(
            "Unable to compare \(String(describing: v1)) and \(String(describing: v2)) at \(path)",
            file: file, line: line)
    }
}

public class UserInputTests: XCTestCase {

    public func testStandardConversion() {
        let chat: [Chat.Message] = [
            .system("You are a useful agent."),
            .user("Tell me a story."),
        ]

        let messages = DefaultMessageGenerator().generate(messages: chat)

        let expected = [
            [
                "role": "system",
                "content": "You are a useful agent.",
            ],
            [
                "role": "user",
                "content": "Tell me a story.",
            ],
        ]

        XCTAssertEqual(expected, messages as? [[String: String]])
    }

    public func testQwen2ConversionText() {
        let chat: [Chat.Message] = [
            .system("You are a useful agent."),
            .user("Tell me a story."),
        ]

        let messages = Qwen2VLMessageGenerator().generate(messages: chat)

        let expected = [
            [
                "role": "system",
                "content": [
                    [
                        "type": "text",
                        "text": "You are a useful agent.",
                    ]
                ],
            ],
            [
                "role": "user",
                "content": [
                    [
                        "type": "text",
                        "text": "Tell me a story.",
                    ]
                ],
            ],
        ]

        assertEqual(expected, messages)
    }

    public func testQwen2ConversionImage() {
        let chat: [Chat.Message] = [
            .system("You are a useful agent."),
            .user(
                "What is this?",
                images: [
                    .url(
                        URL(
                            string: "https://opensource.apple.com/images/projects/mlx.f5c59d8b.png")!
                    )
                ]),
        ]

        let messages = Qwen2VLMessageGenerator().generate(messages: chat)

        let expected = [
            [
                "role": "system",
                "content": [
                    [
                        "type": "text",
                        "text": "You are a useful agent.",
                    ]
                ],
            ],
            [
                "role": "user",
                "content": [
                    [
                        "type": "text",
                        "text": "What is this?",
                    ],
                    [
                        "type": "image"
                    ],
                ],
            ],
        ]

        assertEqual(expected, messages)

        let userInput = UserInput(chat: chat)
        XCTAssertEqual(userInput.images.count, 1)
    }

}
