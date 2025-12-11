// Copyright © 2025 Apple Inc.

import Foundation

struct PresetPrompt: Identifiable {
    let id = UUID()
    let prompt: String
    let enableTools: Bool
    let enableThinking: Bool
    let isLongPrompt: Bool

    init(
        _ prompt: String, enableTools: Bool = false, enableThinking: Bool = false,
        isLongPrompt: Bool = false
    ) {
        self.prompt = prompt
        self.enableTools = enableTools
        self.enableThinking = enableThinking
        self.isLongPrompt = isLongPrompt
    }
}

struct PresetPrompts {
    // Helper to load prompts from markdown files
    private static func loadPrompt(named fileName: String) -> String {
        guard let url = Bundle.main.url(forResource: fileName, withExtension: "md"),
            let content = try? String(contentsOf: url, encoding: .utf8)
        else {
            return "Could not load \(fileName).md. Please ensure it is included in the app bundle."
        }
        return content
    }

    static let all: [PresetPrompt] = [
        PresetPrompt("Why is the sky blue?"),
        PresetPrompt("What would a medieval knight's Yelp review of a dragon's lair look like?"),
        PresetPrompt("Explain why socks disappear in the dryer from the dryer's perspective."),

        PresetPrompt(
            "Write a breaking news report about cats discovering they can vote.",
            enableThinking: true),
        PresetPrompt(
            "Write a performance review for the person whose job is to make sure Mondays feel terrible.",
            enableThinking: true),

        PresetPrompt("What's the weather in Paris?", enableTools: true),
        PresetPrompt("What is the current time?", enableTools: true),

        PresetPrompt(loadPrompt(named: "LongPrompt"), enableThinking: true, isLongPrompt: true),
        PresetPrompt(loadPrompt(named: "CarKeysStory"), isLongPrompt: true),
    ]
}
