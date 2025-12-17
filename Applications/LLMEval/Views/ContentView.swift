// Copyright © 2025 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon
import Metal
import SwiftUI
import Tokenizers

struct ContentView: View {
    @Environment(DeviceStat.self) private var deviceStat

    @State var llm = LLMEvaluator()

    enum DisplayStyle: String, CaseIterable, Identifiable {
        case plain, markdown
        var id: Self { self }
    }

    @State private var selectedDisplayStyle = DisplayStyle.markdown
    @State private var showingPresetPrompts = false
    @State private var isPromptExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header Section
            HeaderView(
                llm: llm,
                selectedDisplayStyle: $selectedDisplayStyle
            )

            Divider()
                .padding(.bottom, 12)

            // Output display
            OutputView(
                output: llm.output,
                displayStyle: selectedDisplayStyle,
                wasTruncated: llm.wasTruncated
            )

            // Prompt input section
            PromptInputView(
                llm: llm,
                isPromptExpanded: $isPromptExpanded,
                showingPresetPrompts: $showingPresetPrompts,
                onGenerate: generate,
                onCancel: cancel
            )

            // Performance Metrics Panel
            MetricsView(
                tokensPerSecond: llm.tokensPerSecond,
                timeToFirstToken: llm.timeToFirstToken,
                promptLength: llm.promptLength,
                totalTokens: llm.totalTokens,
                totalTime: llm.totalTime,
                memoryUsed: deviceStat.gpuUsage.activeMemory,
                cacheMemory: deviceStat.gpuUsage.cacheMemory,
                peakMemory: deviceStat.gpuUsage.peakMemory
            )
        }
        #if os(visionOS)
            .padding(40)
        #else
            .padding()
        #endif
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task {
                        copyToClipboard(llm.output)
                    }
                } label: {
                    Label("Copy Output", systemImage: "doc.on.doc.fill")
                }
                .disabled(llm.output == "")
                .labelStyle(.titleAndIcon)
            }

        }
        .task {
            do {
                // pre-load the weights on launch to speed up the first generation
                _ = try await llm.load()
            } catch {
                llm.output = "Failed: \(error)"
            }
        }
        .sheet(isPresented: $showingPresetPrompts) {
            PresetPromptsSheet(isPresented: $showingPresetPrompts) { preset in
                llm.prompt = preset.prompt
                llm.includeWeatherTool = preset.enableTools
                llm.enableThinking = preset.enableThinking
            }
        }
        .overlay {
            if llm.isLoading {
                LoadingOverlayView(
                    modelInfo: llm.modelInfo,
                    downloadProgress: llm.downloadProgress,
                    progressDescription: llm.totalSize
                )
            }
        }
    }

    private func generate() {
        llm.generate()
    }

    private func cancel() {
        llm.cancelGeneration()
    }

    private func copyToClipboard(_ string: String) {
        #if os(macOS)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(string, forType: .string)
        #else
            UIPasteboard.general.string = string
        #endif
    }
}
