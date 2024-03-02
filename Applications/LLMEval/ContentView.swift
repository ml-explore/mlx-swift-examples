// Copyright © 2024 Apple Inc.

import LLM
import MLX
import Metal
import SwiftUI
import Tokenizers

struct ContentView: View {

    @State var prompt = "compare python and swift"
    @State var llm = LLMEvaluator()

    var body: some View {
        VStack {
            // show the model output
            ScrollView(.vertical) {
                if llm.running {
                    ProgressView()
                }
                Text(llm.output)
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
}

@Observable
class LLMEvaluator {

    @MainActor
    var running = false

    var output = ""

    /// this controls which model loads -- phi4bit is one of the smaller ones so this will fit on
    /// more devices
    let modelConfiguration = ModelConfiguration.phi4bit

    /// parameters controlling the output
    let temperature: Float = 0.0
    let maxTokens = 100

    enum LoadState {
        case idle
        case loaded(LLMModel, LLM.Tokenizer)
    }

    var loadState = LoadState.idle

    /// load and return the model -- can be called multiple times, subsequent calls will
    /// just return the loaded model
    func load() async throws -> (LLMModel, LLM.Tokenizer) {
        switch loadState {
        case .idle:
            let (model, tokenizer) = try await LLM.load(configuration: modelConfiguration) {
                [modelConfiguration] progress in
                DispatchQueue.main.sync {
                    self.output =
                        "Downloading \(modelConfiguration.id): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            loadState = .loaded(model, tokenizer)
            return (model, tokenizer)

        case .loaded(let model, let tokenizer):
            return (model, tokenizer)
        }
    }

    func generate(prompt: String) async {
        do {
            let (model, tokenizer) = try await load()

            await MainActor.run {
                running = true
                self.output = ""
            }

            // augment the prompt as needed
            let prompt = modelConfiguration.prepare(prompt: prompt)
            let promptTokens = MLXArray(tokenizer.encode(text: prompt))

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

            await MainActor.run {
                running = false
            }

        } catch {
            await MainActor.run {
                running = false
                output = "Failed: \(error)"
            }
        }
    }
}
