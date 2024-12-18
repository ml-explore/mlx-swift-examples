// Copyright 2024 Apple Inc.

import Foundation
import SwiftUI
import MLXVLM
import MLX
import CoreImage
import OSLog

struct ContentView: View {
    @State private var image: NSImage?
    @State private var result: String = ""
    @State private var isLoading = false
    @State private var modelContainer: ModelContainer?

        @State var prompt = ""
        @State var llm = VLMEvaluator()
        @Environment(DeviceStat.self) private var deviceStat

    private let imageURL = URL(string: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg")!
    private let prompt = "Describe in detail what you see in the image."

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                if let image {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFit()
                        .cornerRadius(12)
                        .padding()
                        .frame(height: 300)
                }

                if isLoading {
                    ProgressView()
                }
            }
        }
        .task {
            do {
                self.modelContainer = try await VLMModelFactory.shared.loadContainer(
                    configuration: ModelRegistry.qwen2VL2BInstruct4Bit) { progress in
                        let percentage = Int(progress.fractionCompleted * 100)
                        logger.info(" Model loading progress: \(percentage)%")
                    }

                await processImageFromURL()
            } catch {
                logger.error(" Error loading model: \(error.localizedDescription)")
            }
        }
    }
}

extension ContentView {
    private func processImageFromURL() async {
        do {
            let (data, _) = try await URLSession.shared.data(from: imageURL)

            if let uiImage = NSImage(data: data) {
                await MainActor.run {
                    image = uiImage
                    isLoading = true
                }

                if let ciImage = CIImage(data: data) {
                    try await processImage(ciImage)
                }

                await MainActor.run {
                    isLoading = false
                }
            }
        } catch {
            print(" Failed to download image: \(error.localizedDescription)")
        }
    }

    private func processImage(_ ciImage: CIImage) async throws {
        guard let container = modelContainer else { return }

        let prompt = """
          Analyze the image and provide output in valid JSON format with the following structure:
          {
              "description": "detailed description of the image",
              "objects": ["list of main objects"],
              "colors": ["dominant colors"],
              "lighting": "lighting conditions",
              "composition": "compositional analysis"
          }
          
          return response in JSON format {}
          """

        var input = UserInput(prompt: prompt, images: [.ciImage(ciImage)])
        input.processing.resize = .init(width: 448, height: 448)

        let result = try await container.perform { context in
            let input = try await context.processor.prepare(input: input)

            var tokenCount = 0
            var fullResponse = ""

            return try MLXLMCommon.generate(
                input: input,
                parameters: .init(),
                context: context
            ) { tokens in
                tokenCount += 1

                let newText = context.tokenizer.decode(tokens: tokens)
                fullResponse = newText

                Task { @MainActor in
                    self.result = fullResponse
                    // Try to decode JSON as it comes in
                    if let jsonData = fullResponse.data(using: .utf8),
                       let decodedAnalysis = try? JSONDecoder().decode(ImageAnalysis.self, from: jsonData) {
                        self.analysis = decodedAnalysis
                    }
                }

                if tokens.count >= 800 {
                    return .stop
                }
                return .more
            }
        }
    }
}

@Observable
@MainActor
class VLMEvaluator {
    var running = false
    var output = ""
    var modelInfo = ""
    var stat = ""

    let modelConfiguration = ModelRegistry.qwen2VL2BInstruct4Bit
    let generateParameters = MLXLMCommon.GenerateParameters(temperature: 0.6)
    let maxTokens = 800
    let displayEveryNTokens = 4

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    var loadState = LoadState.idle

    func load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let modelContainer = try await VLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) { [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo = "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }

            let numParams = await modelContainer.perform { context in
                context.model.numParameters()
            }

            self.modelInfo = "Loaded \(modelConfiguration.id). Weights: \(numParams / (1024*1024))M"
            loadState = .loaded(modelContainer)
            return modelContainer

        case .loaded(let modelContainer):
            return modelContainer
        }
    }

    func generate(prompt: String, image: CIImage) async {
        guard !running else { return }

        running = true
        self.output = ""

        do {
            let modelContainer = try await load()

            var input = UserInput(prompt: prompt, images: [.ciImage(image)])
            input.processing.resize = .init(width: 448, height: 448)

            let result = try await modelContainer.perform { context in
                let input = try await context.processor.prepare(input: input)

                return try MLXLMCommon.generate(
                    input: input,
                    parameters: generateParameters,
                    context: context
                ) { tokens in
                    if tokens.count % displayEveryNTokens == 0 {
                        let text = context.tokenizer.decode(tokens: tokens)
                        Task { @MainActor in
                            self.output = text
                        }
                    }

                    if tokens.count >= maxTokens {
                        return .stop
                    } else {
                        return .more
                    }
                }
            }

            if result.output != self.output {
                self.output = result.output
            }
            self.stat = " Tokens/second: \(String(format: "%.3f", result.tokensPerSecond))"

        } catch {
            output = "Failed: \(error)"
        }

        running = false
    }
}
