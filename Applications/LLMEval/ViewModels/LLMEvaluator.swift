// Copyright © 2025 Apple Inc.

import Hub
import MLX
import MLXLLM
import MLXLMCommon
import Metal
import SwiftUI

@Observable
@MainActor
class LLMEvaluator {

    var running = false

    var includeWeatherTool = false
    var enableThinking = false
    var maxTokens = 2048

    var prompt = ""
    var output = ""
    var modelInfo = ""

    // Download progress tracking
    var downloadProgress: Double?
    var totalSize: String?

    // Performance metrics
    var tokensPerSecond: Double = 0.0
    var timeToFirstToken: Double = 0.0
    var promptLength: Int = 0
    var totalTokens: Int = 0
    var totalTime: Double = 0.0

    // Track if generation was truncated due to hitting max tokens
    var wasTruncated: Bool = false

    // Timer for tracking TTFT in real-time
    private var ttftTimer: Timer?
    private var generationStartTime: TimeInterval = 0

    // Timer for tracking tokens/sec and total time in real-time
    private var generationTimer: Timer?
    private var firstTokenTime: TimeInterval = 0

    /// This controls which model loads.
    var modelConfiguration = LLMRegistry.qwen3_8b_4bit

    /// Parameters controlling the generation output (max tokens and temperature).
    var generateParameters: GenerateParameters {
        GenerateParameters(maxTokens: maxTokens, temperature: 0.6)
    }

    /// A task responsible for handling the generation process.
    var generationTask: Task<Void, Error>?

    /// Tool executor for function calling
    private let toolExecutor = ToolExecutor()

    enum LoadState {
        case idle
        case loading
        case loaded(ModelContainer)
    }

    var loadState = LoadState.idle

    var isLoading: Bool {
        if case .loading = loadState {
            return true
        }
        return false
    }

    /// Short model name extracted from the full model ID.
    private var modelName: String {
        modelConfiguration.name.components(separatedBy: "/").last ?? modelConfiguration.name
    }

    /// Load and return the model. Can be called multiple times; subsequent calls return the cached model.
    func load() async throws -> ModelContainer {
        while true {
            switch loadState {
            case .idle:
                return try await performLoad()

            case .loading:
                // Already loading, wait and retry
                try await Task.sleep(for: .milliseconds(100))

            case .loaded(let modelContainer):
                return modelContainer
            }
        }
    }

    private func performLoad() async throws -> ModelContainer {
        loadState = .loading
        modelInfo = "Downloading \(modelName)..."
        downloadProgress = 0.0

        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

        let hub = HubApi(
            downloadBase: FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        )

        do {
            let modelDirectory = try await downloadModel(
                hub: hub,
                configuration: modelConfiguration
            ) { [weak self] progress in
                Task { @MainActor in
                    self?.updateDownloadProgress(progress)
                }
            }

            // Verify the download succeeded by checking for model files
            let fileManager = FileManager.default
            let directoryExists = fileManager.fileExists(atPath: modelDirectory.path)
            let contents = (try? fileManager.contentsOfDirectory(atPath: modelDirectory.path)) ?? []
            let hasSafetensors = contents.contains { $0.hasSuffix(".safetensors") }

            if !directoryExists || !hasSafetensors {
                throw NSError(
                    domain: "LLMEvaluator",
                    code: -1,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Model download failed. Please check your network connection and try again."
                    ]
                )
            }

            modelInfo = "Loading \(modelName)..."
            downloadProgress = nil
            totalSize = nil

            let modelContainer = try await LLMModelFactory.shared.loadContainer(
                hub: hub,
                configuration: modelConfiguration
            ) { _ in }

            let numParams = await modelContainer.perform { $0.model.numParameters() }

            self.prompt = PresetPrompts.all[0].prompt
            self.modelInfo = formatModelInfo(name: modelConfiguration.name, parameters: numParams)
            loadState = .loaded(modelContainer)
            return modelContainer

        } catch {
            resetLoadingState()
            throw error
        }
    }

    private func updateDownloadProgress(_ progress: Progress) {
        modelInfo = "Downloading \(modelName) (\(Int(progress.fractionCompleted * 100))%)"
        downloadProgress = progress.fractionCompleted

        // Get file count info
        if progress.totalUnitCount > 0 && progress.totalUnitCount < 100 {
            totalSize = "File \(progress.completedUnitCount + 1) of \(progress.totalUnitCount)"
        } else if progress.totalUnitCount > 0 {
            totalSize =
                "\(formatBytes(progress.completedUnitCount)) of \(formatBytes(progress.totalUnitCount))"
        } else {
            totalSize = nil
        }
    }

    private func resetLoadingState() {
        loadState = .idle
        downloadProgress = nil
        totalSize = nil
    }

    private func formatModelInfo(name: String, parameters: Int) -> String {
        // Extract model name from full ID (e.g., "mlx-community/Qwen3-8B-4bit" -> "Qwen3-8B-4bit")
        let modelName = name.components(separatedBy: "/").last ?? name

        // Format parameter count (convert millions to billions if appropriate)
        let paramMillions = parameters / (1024 * 1024)
        let paramString: String
        if paramMillions >= 1000 {
            let paramBillions = Double(paramMillions) / 1000.0
            paramString = String(format: "%.1fB", paramBillions)
        } else {
            paramString = "\(paramMillions)M"
        }

        return "\(modelName) • \(paramString) parameters"
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }

    private func generate(prompt: String, toolResult: String? = nil) async {
        // Only clear output if this is a fresh generation (not a tool continuation)
        if toolResult == nil {
            self.output = ""
            // Reset metrics at start
            self.totalTokens = 0
            self.tokensPerSecond = 0.0
            self.promptLength = 0
            self.timeToFirstToken = 0.0
            self.totalTime = 0.0
            self.wasTruncated = false

            // Start the real-time TTFT timer
            generationStartTime = Date.timeIntervalSinceReferenceDate
            ttftTimer?.invalidate()
            ttftTimer = Timer.scheduledTimer(withTimeInterval: 0.01, repeats: true) {
                [weak self] _ in
                guard let self = self else { return }
                Task { @MainActor in
                    let elapsed = Date.timeIntervalSinceReferenceDate - self.generationStartTime
                    self.timeToFirstToken = elapsed * 1000  // Convert to ms
                }
            }
        }

        var chat: [Chat.Message] = [
            .system("You are a helpful assistant"),
            .user(prompt),
        ]

        if let toolResult {
            chat.append(.tool(toolResult))
        }

        let userInput = UserInput(
            chat: chat,
            tools: includeWeatherTool ? toolExecutor.allToolSchemas : nil,
            additionalContext: ["enable_thinking": enableThinking]
        )

        do {
            let modelContainer = try await load()

            // Capture parameters on MainActor before entering perform block
            let parameters = generateParameters

            // Seed random generator to ensure varied output each generation
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            try await modelContainer.perform { (context: ModelContext) -> Void in
                let lmInput = try await context.processor.prepare(input: userInput)
                let start = Date.timeIntervalSinceReferenceDate
                let stream = try MLXLMCommon.generate(
                    input: lmInput, parameters: parameters, context: context)

                var iterator = stream.makeAsyncIterator()
                if let first = await iterator.next() {
                    let firstTick = Date.timeIntervalSinceReferenceDate
                    let promptTime = firstTick - start
                    let promptTokenCount = lmInput.text.tokens.size

                    // Update TTFT and prompt length
                    Task { @MainActor in
                        self.ttftTimer?.invalidate()
                        self.ttftTimer = nil
                        self.timeToFirstToken = promptTime * 1000  // Convert to ms
                        self.promptLength = promptTokenCount

                        // Start real-time generation metrics tracking
                        self.firstTokenTime = Date.timeIntervalSinceReferenceDate
                        self.generationTimer?.invalidate()
                        self.generationTimer = Timer.scheduledTimer(
                            withTimeInterval: 0.1, repeats: true
                        ) { [weak self] _ in
                            guard let self = self else { return }
                            Task { @MainActor in
                                let elapsed =
                                    Date.timeIntervalSinceReferenceDate - self.firstTokenTime
                                if elapsed > 0 && self.totalTokens > 0 {
                                    self.tokensPerSecond = Double(self.totalTokens) / elapsed
                                    self.totalTime = elapsed
                                }
                            }
                        }
                    }

                    var generateTokens: Double = 1
                    var pendingToolCall: ToolCall?

                    // Check if first token is a tool call
                    if let toolCall = first.toolCall {
                        pendingToolCall = toolCall
                    } else if let chunk = first.chunk {
                        if !chunk.isEmpty {
                            Task { @MainActor [chunk] in
                                self.output += chunk
                                self.totalTokens += 1
                            }
                        }
                    }

                    // Only continue iterating if we haven't hit a tool call
                    if pendingToolCall == nil {
                        while let next = await iterator.next() {
                            // Check for tool calls
                            if let toolCall = next.toolCall {
                                pendingToolCall = toolCall
                                break
                            }

                            // Handle text chunks
                            if let chunk = next.chunk {
                                if !chunk.isEmpty {
                                    Task { @MainActor [chunk] in
                                        self.output += chunk
                                        self.totalTokens += 1
                                    }
                                    generateTokens += 1
                                }
                            }
                        }
                    }
                    let secondTick = Date.timeIntervalSinceReferenceDate
                    let generateTime = secondTick - firstTick
                    let generateTps = generateTokens / generateTime

                    Task { @MainActor in
                        self.generationTimer?.invalidate()
                        self.generationTimer = nil
                        self.tokensPerSecond = generateTps
                        self.totalTime = generateTime

                        // Check if generation was truncated due to max tokens
                        if self.totalTokens >= parameters.maxTokens ?? Int.max {
                            self.wasTruncated = true
                        }
                    }

                    // Handle tool call if one was made
                    if let toolCall = pendingToolCall {
                        await self.executeToolAndContinue(
                            toolCall: toolCall, originalPrompt: prompt)
                    }
                }
            }

        } catch {
            ttftTimer?.invalidate()
            ttftTimer = nil
            generationTimer?.invalidate()
            generationTimer = nil
            output = "Failed: \(error)"
        }
    }

    private func executeToolAndContinue(toolCall: ToolCall, originalPrompt: String) async {
        // Show tool execution in output
        self.output += "\n\n[Executing tool: \(toolCall.function.name)...]\n\n"

        let result: String
        do {
            result = try await toolExecutor.execute(toolCall)
        } catch {
            result = "Error executing tool: \(error.localizedDescription)"
        }

        // Continue generation with tool result
        await generate(prompt: originalPrompt, toolResult: result)
    }

    func generate() {
        guard !running else { return }

        let currentPrompt = prompt
        guard !currentPrompt.isEmpty else { return }

        generationTask = Task {
            running = true
            await generate(prompt: currentPrompt)
            prompt = ""
            running = false
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        ttftTimer?.invalidate()
        ttftTimer = nil
        generationTimer?.invalidate()
        generationTimer = nil
        running = false
    }
}
