// Copyright © 2025 Apple Inc.

import Foundation
import MLXLMCommon

public typealias ToolSpec = [String: Any]

/// Manages tool definitions and execution for LLM function calling
@MainActor
class ToolExecutor {

    // MARK: - Tool Definitions

    let currentWeatherTool = Tool<WeatherInput, WeatherOutput>(
        name: "get_current_weather",
        description: "Get the current weather in a given location",
        parameters: [
            .required(
                "location", type: .string, description: "The city and state, e.g. San Francisco, CA"
            ),
            .optional(
                "unit",
                type: .string,
                description: "The unit of temperature",
                extraProperties: [
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                ]
            ),
        ]
    ) { input in
        let range = input.unit == "celsius" ? (min: -20.0, max: 40.0) : (min: 0, max: 100)
        let temperature = Double.random(in: range.min ... range.max)
        let conditions = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Stormy"].randomElement()!
        return WeatherOutput(temperature: temperature, conditions: conditions)
    }

    let addTool = Tool<AddInput, AddOutput>(
        name: "add_two_numbers",
        description: "Add two numbers together",
        parameters: [
            .required("first", type: .int, description: "The first number to add"),
            .required("second", type: .int, description: "The second number to add"),
        ]
    ) { input in
        AddOutput(result: input.first + input.second)
    }

    let timeTool = Tool<EmptyInput, TimeOutput>(
        name: "get_time",
        description: "Get the current time",
        parameters: []
    ) { _ in
        TimeOutput(time: Date.now.formatted())
    }

    // MARK: - Tool Execution

    /// Returns all available tool schemas
    var allToolSchemas: [ToolSpec] {
        [currentWeatherTool.schema, addTool.schema, timeTool.schema]
    }

    /// Executes a tool call and returns the result
    func execute(_ toolCall: ToolCall) async throws -> String {
        switch toolCall.function.name {
        case currentWeatherTool.name:
            return try await toolCall.execute(with: currentWeatherTool).toolResult
        case addTool.name:
            return try await toolCall.execute(with: addTool).toolResult
        case timeTool.name:
            return try await toolCall.execute(with: timeTool).toolResult
        default:
            return "Unknown tool: \(toolCall.function.name)"
        }
    }
}
