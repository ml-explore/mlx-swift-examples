// Copyright © 2025 Apple Inc.

import Foundation

// MARK: - Weather Tool

struct WeatherInput: Codable {
    let location: String
    let unit: String?
}

struct WeatherOutput: Codable {
    let temperature: Double
    let conditions: String
}

// MARK: - Add Tool

struct AddInput: Codable {
    let first: Int
    let second: Int
}

struct AddOutput: Codable {
    let result: Int
}

// MARK: - Time Tool

struct EmptyInput: Codable {}

struct TimeOutput: Codable {
    let time: String
}
