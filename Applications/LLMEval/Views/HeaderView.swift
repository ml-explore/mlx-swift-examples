// Copyright © 2025 Apple Inc.

import SwiftUI

struct HeaderView: View {
    @Bindable var llm: LLMEvaluator
    @Binding var selectedDisplayStyle: ContentView.DisplayStyle

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Model info with status
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Model")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text(llm.modelInfo)
                        .font(.headline)
                        .lineLimit(1)
                }

                Spacer()

                if llm.running {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                        Text("Generating...")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            // Controls row
            HStack(spacing: 16) {
                HStack(spacing: 24) {
                    Toggle("Tools", isOn: $llm.includeWeatherTool)
                        .toggleStyle(.switch)
                        .fixedSize()
                        .help("Enable function calling with weather, math, and time tools")

                    Toggle("Thinking", isOn: $llm.enableThinking)
                        .toggleStyle(.switch)
                        .fixedSize()
                        .help("Enable thinking mode (supported by Qwen3)")

                    // Max tokens slider
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Max Tokens: \(llm.maxTokens)")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        Slider(
                            value: Binding(
                                get: { log2(Double(llm.maxTokens)) },
                                set: { llm.maxTokens = Int(pow(2, $0)) }
                            ),
                            in: 10 ... 15,  // 2^10 (1024) to 2^15 (32768)
                            step: 1
                        )
                        .frame(width: 120)
                        .help("Maximum number of tokens to generate (1024-32768)")
                    }
                }

                Spacer()

                Picker("Display", selection: $selectedDisplayStyle) {
                    ForEach(ContentView.DisplayStyle.allCases, id: \.self) { option in
                        Text(option.rawValue.capitalized)
                            .tag(option)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(maxWidth: 180)
            }
        }
        .padding(.bottom, 12)
    }
}
