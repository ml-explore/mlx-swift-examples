//
//  SampleData.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

@MainActor
struct SampleData {
    static let conversation: [Message] = [
        .system("You are a helpful assistant specializing in SwiftUI development."),
        .user("I need help building a weather app in SwiftUI. Where should I start?"),
        .assistant(
            "I'll help you create a weather app! Let's break it down into steps. First, we'll need to design the main view to display current weather conditions. Would you like to start with that?"
        ),
        .user("Yes, that sounds good. What components should I use for the main view?"),
        .assistant(
            "For the main weather view, I recommend using a VStack as the container. You can include:\n\n1. An Image view for the weather icon\n2. Text views for temperature and conditions\n3. HStack for additional metrics like humidity and wind speed"
        ),
        .user("How do I make the UI look modern and polished?"),
        .assistant(
            "To create a modern UI, try these techniques:\n\n- Use SF Symbols for weather icons\n- Add subtle gradients with .background()\n- Include padding and spacing for better layout\n- Implement dark mode support\n\nWould you like to see some example code?"
        ),
        .user("Yes, please show me an example for the main weather view."),
        .assistant(
            "Here's a basic example:\n\nVStack(spacing: 20) {\n    Image(systemName: \"sun.max.fill\")\n        .symbolRenderingMode(.multicolor)\n        .font(.system(size: 64))\n    \n    Text(\"72°\")\n        .font(.largeTitle)\n        .bold()\n    \n    Text(\"Sunny\")\n        .font(.title2)\n}\n.padding()\n.background(.ultraThinMaterial)\n.clipShape(RoundedRectangle(cornerRadius: 20))"
        ),
        .user("That looks great! How would I add animations to this?"),
        .assistant(
            "We can add smooth animations using SwiftUI's animation modifiers. For example:\n\n1. Use withAnimation for state changes\n2. Add .animation() modifier to views\n3. Implement transitions\n\nWould you like to see how to animate the weather changes?"
        ),
    ]
}
