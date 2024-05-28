// Copyright © 2024 Apple Inc.

import LLM
import MLX
import MLXOptimizers
import MLXRandom
import SwiftUI
import Tokenizers

struct ContentView: View {

    @State var evaluator = LoRAEvaluator()

    @State var prompt = """
        table: 1-10015132-16
        columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team
        Q: What is terrence ross' nationality
        A:
        """

    var body: some View {
        VStack {
            HStack {
                if let progress = evaluator.progress {
                    if let current = progress.current, let limit = progress.limit {
                        ProgressView(progress.title, value: current, total: limit)
                    } else {
                        ProgressView(progress.title)
                    }
                }
            }
            .frame(maxWidth: .infinity, minHeight: 25)

            VStack {
                ScrollView(.vertical) {
                    ScrollViewReader { sp in
                        Group {
                            Text(evaluator.output)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity)
                        }
                        .onChange(of: evaluator.output) { _, _ in
                            sp.scrollTo("bottom")
                        }
                        .padding()

                        Spacer()
                            .frame(width: 1, height: 1)
                            .id("bottom")
                    }
                }

                // controls for each of the different states
                VStack {
                    switch evaluator.state {
                    case .idle:
                        Button("Start", action: start)

                    case .training:
                        EmptyView()

                    case .evaluate:
                        Group {
                            TextEditor(text: $prompt)
                                .frame(minHeight: 60)
                            Button("Evaluate", action: evaluate)
                        }
                        .disabled(evaluator.progress != nil)

                    case .failed(let message):
                        Text("Failed: \(message)")
                            .bold()
                            .foregroundStyle(.red)
                    }
                }
                .padding()
                .frame(maxWidth: .infinity)
            }
        }
        .padding()
    }

    func start() {
        Task {
            await evaluator.start()
        }
    }

    func evaluate() {
        Task {
            await evaluator.evaluate(prompt: prompt)
        }
    }
}

/// Progress reporting with a title.
struct Progress: Equatable {
    let title: String
    let current: Double?
    let limit: Double?
}

@Observable
class LoRAEvaluator {

    enum State {
        case idle
        case training
        case evaluate
        case failed(String)
    }

    enum ModelState {
        case idle
        case loaded(LLMModel, Tokenizer)
    }

    var state = State.idle
    var progress: Progress?

    var output = ""

    private let modelConfiguration = ModelConfiguration.mistral7B4bit
    private var model: ModelState = .idle

    private let loraLayers = 4
    private let learningRate: Float = 1e-5
    private let parameters = LoRATrain.Parameters(batchSize: 1, iterations: 200)

    private let generateParameters = GenerateParameters(temperature: 0.6, topP: 0.9)
    private let evaluateShowEvery = 8
    private let maxTokens = 200

    private func loadModel() async throws -> (LLMModel, Tokenizer) {
        switch self.model {
        case .idle:
            let name = modelConfiguration.name
            await MainActor.run {
                progress = .init(title: "Loading \(name)", current: 0, limit: 1)
            }

            let (model, tokenizer) = try await LLM.load(configuration: modelConfiguration) {
                progress in
                if progress.fractionCompleted < 1.0 {
                    DispatchQueue.main.sync {
                        self.progress = .init(
                            title: "Download \(name)", current: progress.fractionCompleted,
                            limit: 1.0)
                    }
                }
            }
            eval(model)
            self.model = .loaded(model, tokenizer)
            return (model, tokenizer)

        case .loaded(let model, let tokenizer):
            return (model, tokenizer)
        }
    }

    private func loadLoRAData(name: String) throws -> [String]? {
        if let url = Bundle.main.url(forResource: name, withExtension: "jsonl") {
            return try LLM.loadLoRAData(url: url)
        }
        return nil
    }

    func start() async {
        do {
            try await startInner()
        } catch {
            self.state = .failed("Failed: \(error)")
        }
    }

    private func startInner() async throws {
        // setup
        GPU.set(cacheLimit: 32 * 1024 * 1024)
        await MainActor.run {
            output = ""
            state = .training
        }

        // load the model
        let (model, tokenizer) = try await loadModel()

        // apply LoRA adapters and train
        guard let layerProvider = model as? LoRAModel else {
            state = .failed("Model must implement the LoRALayerProvider protocol")
            return
        }
        LoRATrain.convert(
            model: model, layers: Array(layerProvider.loraLinearLayers().suffix(loraLayers)))

        let train = try loadLoRAData(name: "train")
        let valid = try loadLoRAData(name: "valid")
        guard let train, let valid else {
            state = .failed("Failed to load train/validation data")
            return
        }

        let optimizer = Adam(learningRate: learningRate)
        try await LoRATrain.train(
            model: model, train: train, validate: valid, optimizer: optimizer, tokenizer: tokenizer,
            parameters: parameters
        ) { progress in
            await MainActor.run {
                switch progress {
                case .train(let i, _, _, _):
                    self.progress = .init(
                        title: "Train", current: Double(i), limit: Double(parameters.iterations))
                case .validation:
                    output += "\n"
                default:
                    break
                }

                output += progress.description + "\n"
            }

            return .more
        }

        // done training, test
        await MainActor.run {
            self.progress = .init(title: "Testing", current: nil, limit: nil)
        }
        guard let test = try loadLoRAData(name: "test") else {
            state = .failed("Failed to load test data")
            return
        }

        let loss = LoRATrain.evaluate(
            model: model, dataset: test, tokenizer: tokenizer, batchSize: 1, batchCount: 0)
        await MainActor.run {
            self.progress = nil
            self.output += "\n"
            self.output += "Test loss \(loss.formatted()), ppl \(exp(loss).formatted())\n"
            self.state = .evaluate
        }
    }

    func evaluate(prompt: String) async {
        do {
            try await evaluateInner(prompt: prompt)
        } catch {
            self.state = .failed("Failed: \(error)")
        }
    }

    func evaluateInner(prompt: String) async throws {
        await MainActor.run {
            self.progress = .init(title: "Evaluating", current: nil, limit: nil)
            self.output = ""
        }

        MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

        let (model, tokenizer) = try await loadModel()

        // prepare the prompt
        let preparedPrompt = modelConfiguration.prepare(prompt: prompt)
        let promptTokens = tokenizer.encode(text: preparedPrompt)

        // evaluate
        let result = await LLM.generate(
            promptTokens: promptTokens, parameters: generateParameters, model: model,
            tokenizer: tokenizer,
            extraEOSTokens: modelConfiguration.extraEOSTokens,
            didGenerate: { tokens in
                if tokens.count % evaluateShowEvery == 0 {
                    let fullOutput = tokenizer.decode(tokens: tokens)
                    await MainActor.run {
                        self.output = fullOutput
                    }
                }
                return tokens.count >= maxTokens ? .stop : .more
            })

        await MainActor.run {
            self.output = result.output
            self.progress = nil
        }
    }
}
