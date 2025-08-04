// Copyright © 2025 Apple Inc.

import Foundation

// Extension on Codable to handle JSON encoding with snake case
extension Encodable {
    public var toolResult: String {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase

        guard let data = try? encoder.encode(self) else { return "{}" }
        return String(data: data, encoding: .utf8) ?? "{}"
    }
}
