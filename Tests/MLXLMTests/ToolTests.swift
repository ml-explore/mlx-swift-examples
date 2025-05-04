import Foundation
import MLXLMCommon
import Testing

struct WeatherInput: Codable {
    let location: String
    let unit: String
}

struct WeatherOutput: Codable {
    let temperature: Double
    let conditions: String
}

struct ToolSchemaTests {
    @Test("Test Weather Tool Schema Generation")
    func testWeatherToolSchemaGeneration() throws {
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
}
