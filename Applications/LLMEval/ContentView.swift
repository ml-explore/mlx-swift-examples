// Copyright © 2024 Apple Inc.

import LLM
import MLX
import MLXRandom
import MarkdownUI
import Metal
import SwiftUI
import Tokenizers

struct ContentView: View {

    @State var prompt = "compare python and swift"
    @State var llm = LLMEvaluator()

    enum displayStyle: String, CaseIterable, Identifiable {
        case plain, markdown
        var id: Self { self }
    }

    @State private var selectedDisplayStyle = displayStyle.markdown

    var body: some View {
        VStack(alignment: .leading) {
            VStack {
                HStack {
                    Text(llm.modelInfo)
                        .textFieldStyle(.roundedBorder)

                    Spacer()

                    Text(llm.stat)
                }
                HStack {
                    Spacer()
                    if llm.running {
                        ProgressView()
                            .frame(maxHeight: 20)
                        Spacer()
                    }
                    Picker("", selection: $selectedDisplayStyle) {
                        ForEach(displayStyle.allCases, id: \.self) { option in
                            Text(option.rawValue.capitalized)
                                .tag(option)
                        }

                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 150)
                }
            }

            // show the model output
            ScrollView(.vertical) {
                ScrollViewReader { sp in
                    Group {
                        if selectedDisplayStyle == .plain {
                            Text(llm.output)
                                .textSelection(.enabled)
                        } else {
                            Markdown(llm.output)
                                .textSelection(.enabled)
                        }
                    }
                    .onChange(of: llm.output) { _, _ in
                        sp.scrollTo("bottom")
                    }

                    Spacer()
                        .frame(width: 1, height: 1)
                        .id("bottom")
                }
            }

            HStack {
                TextField("prompt", text: $prompt)
                    .onSubmit(generate)
                    .disabled(llm.running)
                Button("generate", action: generate)
                    .disabled(llm.running)
            }
        }
        .padding()
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
            // pre-load the weights on launch to speed up the first generation
            _ = try? await llm.load()
        }
    }

    private func generate() {
        Task {
            await llm.generate(prompt: prompt)
        }
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

@Observable
class LLMEvaluator {

    @MainActor
    var running = false

    var output = ""
    var modelInfo = ""
    var stat = ""

    /// this controls which model loads -- phi4bit is one of the smaller ones so this will fit on
    /// more devices
    let modelConfiguration = ModelConfiguration.phi4bit

    /// parameters controlling the output
    let temperature: Float = 0.6
    let maxTokens = 240

    enum LoadState {
        case idle
        case loaded(LLMModel, Tokenizers.Tokenizer)
    }

    var loadState = LoadState.idle

    /// load and return the model -- can be called multiple times, subsequent calls will
    /// just return the loaded model
    func load() async throws -> (LLMModel, Tokenizers.Tokenizer) {
        switch loadState {
        case .idle:
            // limit the buffer cache
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let (model, tokenizer) = try await LLM.load(configuration: modelConfiguration) {
                [modelConfiguration] progress in
                DispatchQueue.main.sync {
                    self.modelInfo =
                        "Downloading \(modelConfiguration.id): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            self.modelInfo =
                "Loaded \(modelConfiguration.id).  Weights: \(MLX.GPU.activeMemory / 1024 / 1024)M"
            loadState = .loaded(model, tokenizer)
            return (model, tokenizer)

        case .loaded(let model, let tokenizer):
            return (model, tokenizer)
        }
    }

    func generate(prompt: String) async {
        let startTime = Date()
        do {
            let (model, tokenizer) = try await load()

            await MainActor.run {
                running = true
                self.output = ""
            }

            // augment the prompt as needed
            let prompt = modelConfiguration.prepare(prompt: prompt)
            let promptTokens = MLXArray(tokenizer.encode(text: prompt))

            let initTime = Date()
            let initDuration = initTime.timeIntervalSince(startTime)
            await MainActor.run {
                self.stat = "Init: \(String(format: "%.3f", initDuration))s"
            }

            // each time you generate you will get something new
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            var outputTokens = [Int]()

            for token in TokenIterator(prompt: promptTokens, model: model, temp: temperature) {
                let tokenId = token.item(Int.self)

                if tokenId == tokenizer.unknownTokenId || tokenId == tokenizer.eosTokenId {
                    break
                }

                outputTokens.append(tokenId)
                let text = tokenizer.decode(tokens: outputTokens)

                // update the output -- this will make the view show the text as it generates
                await MainActor.run {
                    self.output = text
                }

                if outputTokens.count == maxTokens {
                    break
                }
            }

            let tokenDuration = Date().timeIntervalSince(initTime)
            let tokensPerSecond = Double(outputTokens.count) / tokenDuration

            await MainActor.run {
                running = false
                self.stat += " Token/second: \(String(format: "%.3f", tokensPerSecond))"
            }

        } catch {
            await MainActor.run {
                running = false
                output = "Failed: \(error)"
            }
        }
    }
}
