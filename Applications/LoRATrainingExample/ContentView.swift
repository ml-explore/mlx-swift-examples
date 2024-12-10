// Copyright © 2024 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
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
struct Progress: Equatable, Sendable {
    let title: String
    let current: Double?
    let limit: Double?
}

@Observable
@MainActor
class LoRAEvaluator {

    enum State: Sendable {
        case idle
        case training
        case evaluate
        case failed(String)
    }

    enum ModelState: Sendable {
        case idle
        case loaded(ModelContainer)
    }

    var state = State.idle
    var progress: Progress?

    var output = ""

    private let modelConfiguration = ModelRegistry.mistral7B4bit
    private var model: ModelState = .idle

    private let loraLayers = 4
    private let learningRate: Float = 1e-5
    private let parameters = LoRATrain.Parameters(batchSize: 1, iterations: 200)

    private let generateParameters = GenerateParameters(temperature: 0.6, topP: 0.9)
    private let evaluateShowEvery = 8
    private let maxTokens = 200

    private func loadModel() async throws -> ModelContainer {
        switch self.model {
        case .idle:
            let name = modelConfiguration.name
            await MainActor.run {
                progress = .init(title: "Loading \(name)", current: 0, limit: 1)
            }

            let modelContainer = try await LLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) {
                progress in
                Task { @MainActor in
                    self.progress = .init(
                        title: "Download \(name)", current: progress.fractionCompleted,
                        limit: 1.0)
                }
            }
            self.model = .loaded(modelContainer)
            return modelContainer

        case .loaded(let modelContainer):
            return modelContainer
        }
    }

    private func loadLoRAData(name: String) throws -> [String]? {
        if let url = Bundle.main.url(forResource: name, withExtension: "jsonl") {
            return try MLXLLM.loadLoRAData(url: url)
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

    nonisolated private func loraLayers(model: Module) -> LoRALinearLayers {
        guard let layerProvider = model as? LoRAModel else {
            // the layerProvider will indicate which Linear layers need to be replaced
            fatalError(
                "Model \(type(of: model)) (\(modelConfiguration.name)) must implement the LoRALayerProvider protocol"
            )
        }

        return Array(layerProvider.loraLinearLayers().suffix(loraLayers))
    }

    private func startInner() async throws {
        // setup
        GPU.set(cacheLimit: 32 * 1024 * 1024)
        await MainActor.run {
            output = ""
            state = .training
        }

        // load the model
        let modelContainer = try await loadModel()

        // apply LoRA adapters and train
        await modelContainer.perform { context in
            LoRATrain.convert(
                model: context.model, layers: loraLayers(model: context.model))
        }

        let train = try loadLoRAData(name: "train")
        let valid = try loadLoRAData(name: "valid")
        guard let train, let valid else {
            state = .failed("Failed to load train/validation data")
            return
        }

        try await modelContainer.perform { context in
            let optimizer = Adam(learningRate: learningRate)
            try LoRATrain.train(
                model: context.model, train: train, validate: valid, optimizer: optimizer,
                tokenizer: context.tokenizer,
                parameters: parameters
            ) { progress in
                Task { @MainActor in
                    switch progress {
                    case .train(let i, _, _, _):
                        self.progress = .init(
                            title: "Train", current: Double(i), limit: Double(parameters.iterations)
                        )
                    case .validation:
                        output += "\n"
                    default:
                        break
                    }
                    output += progress.description + "\n"
                }

                return .more
            }
        }

        // done training, test
        self.progress = .init(title: "Testing", current: nil, limit: nil)
        guard let test = try loadLoRAData(name: "test") else {
            state = .failed("Failed to load test data")
            return
        }

        let loss = await modelContainer.perform { context in
            LoRATrain.evaluate(
                model: context.model, dataset: test,
                tokenizer: context.tokenizer, batchSize: 1, batchCount: 0)
        }

        self.progress = nil
        self.output += "\n"
        self.output += "Test loss \(loss.formatted()), ppl \(exp(loss).formatted())\n"
        self.state = .evaluate
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

        let modelContainer = try await loadModel()

        // evaluate
        let result = try await modelContainer.perform { context in
            let input = try await context.processor.prepare(input: .init(prompt: prompt))
            return try MLXLMCommon.generate(
                input: input, parameters: generateParameters, context: context
            ) { tokens in
                if tokens.count % evaluateShowEvery == 0 {
                    let fullOutput = context.tokenizer.decode(tokens: tokens)
                    Task { @MainActor in
                        self.output = fullOutput
                    }
                }
                return tokens.count >= maxTokens ? .stop : .more
            }
        }

        self.output = result.output
        self.progress = nil
    }
}
