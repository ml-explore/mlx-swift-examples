// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

// MARK: - Time Tool

struct EmptyInput: Codable {}

struct TimeOutput: Codable {
    let time: String
}

/// Simple tool integration for testing.
let timeTool = Tool<EmptyInput, TimeOutput>(
    name: "get_time",
    description: "Get the current time",
    parameters: []
) { _ in
    TimeOutput(time: Date.now.formatted())
}
