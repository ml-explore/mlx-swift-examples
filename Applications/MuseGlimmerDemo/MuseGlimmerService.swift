import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Tokenizers

/// What the model is doing between "Send" and the first token.
///
/// Worth surfacing because the wait is long and entirely front-loaded: an image
/// expands into up to 4096 prompt tokens, and every one of them has to be
/// prefilled through the 52-layer text stack at roughly 200 tokens/s before any
/// output can appear. Without this the app looks hung.
enum GenerationEvent: Sendable {
    /// Prompt has been tokenized; `image` of `total` tokens came from the image.
    case prompt(total: Int, image: Int)
    /// `processed` of `total` prompt positions submitted to the GPU.
    case prefill(processed: Int, total: Int)
    case chunk(String)
    case completed(promptTime: TimeInterval, tokensPerSecond: Double, peakBytes: Int)
    case failed(String)
}

/// Loads Muse-Glimmer once and streams generations from it.
///
/// Everything after the initial download runs on-device; there are no network
/// calls on the generate path.
@MainActor
@Observable
final class MuseGlimmerService {

    /// Roughly 20 GB resident for the 4-bit weights, of which ~3.7 GB is the
    /// unquantized bf16 vision tower.
    static let configuration = VLMRegistry.museGlimmer30B4bit

    enum LoadState {
        case idle
        case loading(Progress?)
        case ready
        case failed(String)
    }

    private(set) var loadState: LoadState = .idle

    private var container: ModelContainer?

    /// Tunes the allocator for a ~20 GB resident model rather than the ~2 GB the
    /// example apps assume. The default 20 MB buffer cache thrashes badly at this
    /// size, since a single image encode churns multi-hundred-MB activations.
    static func configureMemory() {
        Memory.cacheLimit = 2 * 1024 * 1024 * 1024
        Memory.memoryLimit = 48 * 1024 * 1024 * 1024
    }

    func load() async {
        if case .ready = loadState { return }
        if case .loading = loadState { return }

        Self.configureMemory()
        loadState = .loading(nil)
        do {
            let container = try await loadModelContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: Self.configuration
            ) { progress in
                Task { @MainActor in
                    self.loadState = .loading(progress)
                }
            }
            self.container = container
            loadState = .ready
        } catch {
            loadState = .failed(String(describing: error))
        }
    }

    /// Streams a response for `prompt` with an optional image attached, reporting
    /// prompt composition and prefill progress before the first token.
    /// `maxImageTokens` caps how many merged vision tokens the image may become.
    /// It is the strongest latency lever available: each one becomes a prompt
    /// token that must be prefilled, so halving the budget roughly halves
    /// time-to-first-token. Passed through the model-agnostic `maxPixels` hook,
    /// where one merged token covers 28x28 pixels.
    func generate(
        prompt: String,
        image: URL?,
        maxTokens: Int,
        maxImageTokens: Int
    ) throws -> AsyncStream<GenerationEvent> {
        guard let container else {
            throw MuseGlimmerServiceError.notLoaded
        }

        let (stream, continuation) = AsyncStream<GenerationEvent>.makeStream()

        let work = Task {
            do {
                try await container.perform { (context: ModelContext) in
                    // `UserInput` is not Sendable, so it is built inside the
                    // container's isolation rather than captured across it.
                    let images: [UserInput.Image] = image.map { [.url($0)] } ?? []
                    let chat = [
                        Chat.Message(role: .user, content: prompt, images: images, videos: [])
                    ]
                    let processing = UserInput.Processing(
                        maxPixels: maxImageTokens * 28 * 28)
                    let lmInput = try await context.processor.prepare(
                        input: UserInput(chat: chat, processing: processing))

                    let imageTokenId =
                        (context.model as? MuseGlimmer)?.config.imageTokenId ?? 200_092
                    let tokens = lmInput.text.tokens.asArray(Int.self)
                    continuation.yield(
                        .prompt(
                            total: tokens.count,
                            image: tokens.count { $0 == imageTokenId }))

                    // Greedy, so the app's output can be compared against the
                    // Python reference.
                    var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
                    // Chunks are pipelined with asyncEval, so this reports graph
                    // submission and runs slightly ahead of GPU completion. Fine
                    // for a progress bar; don't read it as a hard timing signal.
                    parameters.prefill = PrefillParameters(progress: { processed, total in
                        continuation.yield(.prefill(processed: processed, total: total))
                    })

                    for await generation in try MLXLMCommon.generate(
                        input: lmInput, parameters: parameters, context: context)
                    {
                        switch generation {
                        case .chunk(let text):
                            continuation.yield(.chunk(text))
                        case .info(let info):
                            continuation.yield(
                                .completed(
                                    promptTime: info.promptTime,
                                    tokensPerSecond: info.tokensPerSecond,
                                    peakBytes: Memory.snapshot().peakMemory))
                        default:
                            break
                        }
                    }
                }
            } catch is CancellationError {
                // Stopped by the user; whatever streamed already stays on screen.
            } catch {
                continuation.yield(.failed(String(describing: error)))
            }
            continuation.finish()
        }

        // Cancelling the consumer has to cancel the generation, or a 30B forward
        // keeps running with nobody reading it.
        continuation.onTermination = { _ in work.cancel() }
        return stream
    }
}

enum MuseGlimmerServiceError: LocalizedError {
    case notLoaded

    var errorDescription: String? {
        switch self {
        case .notLoaded: "The model is not loaded yet."
        }
    }
}
