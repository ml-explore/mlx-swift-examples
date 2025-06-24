import Foundation
import MLXLMCommon
import Testing

struct ToolTests {
    @Test("Test Weather Tool Schema Generation")
    func testWeatherToolSchemaGeneration() throws {
        struct WeatherInput: Codable {
            let location: String
            let unit: String?
        }

        struct WeatherOutput: Codable {
            let temperature: Double
            let conditions: String
        }

        let tool = Tool<WeatherInput, WeatherOutput>(
            name: "get_current_weather",
            description: "Get the current weather in a given location",
            parameters: [
                .required(
                    "location", type: .string, description: "The city, e.g. Istanbul"
                ),
                .optional(
                    "unit",
                    type: .string,
                    description: "The unit of temperature",
                    extraProperties: [
                        "enum": ["celsius", "fahrenheit"]
                    ]
                ),
            ]
        ) { input in
            WeatherOutput(temperature: 14.0, conditions: "Sunny")
        }

        let actual = tool.schema as NSDictionary

        let expected: NSDictionary = [
            "type": "function",
            "function": [
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "location": [
                            "type": "string",
                            "description": "The city, e.g. Istanbul",
                        ],
                        "unit": [
                            "type": "string",
                            "description": "The unit of temperature",
                            "enum": ["celsius", "fahrenheit"],
                        ],
                    ],
                    "required": ["location"],
                ],
            ],
        ]

        #expect(actual == expected)
    }

    @Test("Test Tool Call Detection in Generated Text")
    func testToolCallDetection() throws {
        let processor = ToolCallProcessor()
        let chunks: [String] = [
            "<tool", "_", "call>", "{", "\"", "name", "\"", ":", " ", "\"", "get", "_", "current",
            "_", "weather", "\"", ",", " ", "\"", "arguments", "\"", ":", " ", "{", "\"",
            "location", "\"", ":", " ", "\"", "San", " Francisco", "\"", ",", " ", "\"", "unit",
            "\"", ":", " ", "\"", "celsius", "\"", "}", "}", "</tool", "_", "call>",
        ]

        for chunk in chunks {
            let result = processor.processChunk(chunk)
            #expect(result == nil)
        }

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)

        #expect(toolCall.function.name == "get_current_weather")
        #expect(toolCall.function.arguments["location"] == .string("San Francisco"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }
}
