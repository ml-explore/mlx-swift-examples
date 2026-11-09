import AppKit
import Foundation
import MLX
import MLXLMCommon
import SwiftUI
import UniformTypeIdentifiers

/// File scope rather than a `View` static: it is read from a Sendable closure in
/// the paste handler, which cannot touch main-actor-isolated state.
private let imageTypes: Set<String> = [
    "png", "jpg", "jpeg", "heic", "heif", "tiff", "tif", "webp", "gif", "bmp",
]

struct ContentView: View {
    @State private var service = MuseGlimmerService()

    @State private var prompt = "Describe this image."
    @State private var droppedImage: URL?
    @State private var thumbnail: NSImage?
    @State private var isTargeted = false

    @State private var output = ""
    @State private var isGenerating = false
    @State private var stats: String?
    @State private var errorMessage: String?
    @State private var maxTokens = 400

    /// Well below the checkpoint's 4096. At 4096 a large image costs ~30 s before
    /// the first token, almost all of it prefilling the ~4100 prompt tokens the
    /// image expands into; 1024 lands around 2-3 s with little visible loss.
    @State private var maxImageTokens = 1024
    @State private var generateTask: Task<Void, Never>?

    /// Where we are between "Send" and the first token. The wait is long and
    /// entirely front-loaded, so it needs to be visible.
    enum Phase: Equatable {
        case idle
        case preparing
        case encodingImage
        case prefilling
        /// Prefill is done but nothing is streaming yet. The framework drops
        /// reasoning from the public `Generation` stream, and Muse-Glimmer reasons
        /// at length before answering, so this window is silent — several seconds
        /// on top of prefill with no output at all unless it is named.
        case reasoning
        case streaming
    }

    @State private var phase: Phase = .idle
    @State private var promptComposition: (total: Int, image: Int)?
    @State private var prefill: (processed: Int, total: Int)?
    @State private var timeToFirstToken: TimeInterval?

    var body: some View {
        VStack(spacing: 12) {
            header

            HSplitView {
                imageWell
                    .frame(minWidth: 240, idealWidth: 300)
                outputPane
                    .frame(minWidth: 340)
            }

            promptBar
        }
        .padding(14)
        .task {
            // An image path passed on the command line preloads the well, which
            // makes the drop-to-describe flow scriptable for smoke tests.
            if let path = CommandLine.arguments.dropFirst().first,
                imageTypes.contains((path as NSString).pathExtension.lowercased())
            {
                setImage(URL(fileURLWithPath: path))
            }
            await service.load()
        }
        .onDisappear { generateTask?.cancel() }
    }

    // MARK: - Header

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 10) {
            Text("Muse-Glimmer 30B")
                .font(.headline)
            Text("on-device")
                .font(.caption)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.quaternary, in: Capsule())

            Spacer()

            switch service.loadState {
            case .idle:
                Text("Idle").foregroundStyle(.secondary).font(.caption)
            case .loading(let progress):
                HStack(spacing: 6) {
                    if let progress, progress.totalUnitCount > 0 {
                        ProgressView(value: progress.fractionCompleted)
                            .frame(width: 120)
                        Text("\(Int(progress.fractionCompleted * 100))%")
                            .font(.caption.monospacedDigit())
                    } else {
                        ProgressView().controlSize(.small)
                        Text("Loading…").font(.caption)
                    }
                }
                .foregroundStyle(.secondary)
            case .ready:
                Label("Ready", systemImage: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
            case .failed(let message):
                Label("Load failed", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .help(message)
            }
        }
    }

    // MARK: - Image well

    @ViewBuilder
    private var imageWell: some View {
        VStack(spacing: 8) {
            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(
                        isTargeted ? Color.accentColor : Color.secondary.opacity(0.4),
                        style: StrokeStyle(lineWidth: isTargeted ? 2 : 1, dash: [6, 4])
                    )
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(isTargeted ? Color.accentColor.opacity(0.08) : Color.clear))

                if let thumbnail {
                    Image(nsImage: thumbnail)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .padding(6)
                } else {
                    VStack(spacing: 6) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 28))
                            .foregroundStyle(.secondary)
                        Text("Drop an image").font(.callout).foregroundStyle(.secondary)
                        Text("or click to choose").font(.caption2).foregroundStyle(.tertiary)
                    }
                }
            }
            .frame(minHeight: 240)
            .contentShape(Rectangle())
            .onTapGesture(perform: chooseImage)
            .dropDestination(for: URL.self) { urls, _ in
                guard
                    let url = urls.first(where: {
                        imageTypes.contains($0.pathExtension.lowercased())
                    })
                else { return false }
                setImage(url)
                return true
            } isTargeted: {
                isTargeted = $0
            }
            .onPasteCommand(of: [.fileURL]) { providers in
                for provider in providers {
                    _ = provider.loadObject(ofClass: URL.self) { url, _ in
                        guard let url,
                            imageTypes.contains(url.pathExtension.lowercased())
                        else { return }
                        Task { @MainActor in setImage(url) }
                    }
                }
            }

            if let droppedImage {
                HStack(spacing: 6) {
                    Text(droppedImage.lastPathComponent)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .layoutPriority(1)
                    Button("Clear") {
                        self.droppedImage = nil
                        self.thumbnail = nil
                    }
                    .buttonStyle(.link)
                    .font(.caption)
                    .fixedSize()
                }
            }
        }
    }

    // MARK: - Output

    /// Explains the pre-token wait, which is almost all of the perceived latency.
    ///
    /// An image expands into up to 4096 prompt tokens, and prefill runs at roughly
    /// 200 tokens/s through the 52-layer text stack, so a large image can mean
    /// tens of seconds before anything appears. Showing the token count alongside
    /// the progress bar makes the cost legible rather than mysterious.
    @ViewBuilder
    private var statusStrip: some View {
        if phase != .idle || timeToFirstToken != nil {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    switch phase {
                    case .idle:
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text("Done")
                    case .preparing:
                        ProgressView().controlSize(.small)
                        Text("Preparing prompt…")
                    case .encodingImage:
                        ProgressView().controlSize(.small)
                        Text("Resizing and encoding image…")
                    case .prefilling:
                        ProgressView().controlSize(.small)
                        if let prefill, prefill.total > 0 {
                            Text("Prefilling \(prefill.processed) / \(prefill.total) tokens")
                        } else {
                            Text("Prefilling prompt…")
                        }
                    case .reasoning:
                        ProgressView().controlSize(.small)
                        Text("Reasoning (not streamed by the framework)…")
                    case .streaming:
                        Image(systemName: "text.cursor")
                        Text("Generating")
                    }

                    Spacer()

                    if let timeToFirstToken {
                        Text(String(format: "first token %.1fs", timeToFirstToken))
                            .foregroundStyle(.secondary)
                    }
                }
                .font(.caption)

                if phase == .prefilling, let prefill, prefill.total > 0 {
                    ProgressView(
                        value: Double(prefill.processed), total: Double(prefill.total))
                }

                if let promptComposition {
                    Text(
                        promptComposition.image > 0
                            ? "\(promptComposition.total) prompt tokens "
                                + "(\(promptComposition.image) from the image)"
                            : "\(promptComposition.total) prompt tokens"
                    )
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                }
            }
            .padding(.horizontal, 2)
        }
    }

    @ViewBuilder
    private var outputPane: some View {
        VStack(alignment: .leading, spacing: 6) {
            statusStrip

            ScrollView {
                // The framework strips the protocol's control tokens and withholds
                // reasoning, so what arrives here is the answer text alone.
                Text(output.isEmpty ? "Response will stream here." : output)
                    .font(.body)
                    .foregroundStyle(output.isEmpty ? .tertiary : .primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
            }
            .background(.quinary, in: RoundedRectangle(cornerRadius: 8))

            if let errorMessage {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
            if let stats {
                Text(stats)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Prompt bar

    @ViewBuilder
    private var promptBar: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                TextField("Prompt", text: $prompt, axis: .vertical)
                    .lineLimit(1 ... 3)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(start)

                if isGenerating {
                    Button("Stop") {
                        generateTask?.cancel()
                    }
                    .keyboardShortcut(".", modifiers: .command)
                } else {
                    Button("Send", action: start)
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(!isReady || prompt.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }

            HStack(spacing: 16) {
                HStack(spacing: 6) {
                    Text("Max tokens")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Slider(
                        value: .init(
                            get: { Double(maxTokens) }, set: { maxTokens = Int($0) }),
                        in: 64 ... 2048, step: 64)
                    Text("\(maxTokens)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 42, alignment: .trailing)
                }

                HStack(spacing: 6) {
                    Text("Image budget")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Slider(
                        value: .init(
                            get: { Double(maxImageTokens) },
                            set: { maxImageTokens = Int($0) }),
                        in: 256 ... 4096, step: 256)
                    Text("\(maxImageTokens)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 42, alignment: .trailing)
                }
                .help(
                    "Caps how many tokens the image becomes. The dominant cost "
                        + "before the first token.")
            }
        }
    }

    private var isReady: Bool {
        if case .ready = service.loadState { return true }
        return false
    }

    // MARK: - Actions

    private func chooseImage() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK, let url = panel.url {
            setImage(url)
        }
    }

    private func setImage(_ url: URL) {
        droppedImage = url
        thumbnail = NSImage(contentsOf: url)
    }

    private func start() {
        guard isReady, !isGenerating else { return }
        let prompt = prompt
        let image = droppedImage
        let maxTokens = maxTokens
        let maxImageTokens = maxImageTokens

        output = ""
        stats = nil
        errorMessage = nil
        isGenerating = true
        phase = image == nil ? .preparing : .encodingImage
        promptComposition = nil
        prefill = nil
        timeToFirstToken = nil
        let started = Date()

        generateTask = Task {
            do {
                for await event in try service.generate(
                    prompt: prompt, image: image, maxTokens: maxTokens,
                    maxImageTokens: maxImageTokens)
                {
                    if Task.isCancelled { break }
                    switch event {
                    case .prompt(let total, let imageTokens):
                        await MainActor.run {
                            promptComposition = (total, imageTokens)
                            phase = .prefilling
                        }
                    case .prefill(let processed, let total):
                        await MainActor.run {
                            prefill = (processed, total)
                            // A terminal (total, total) means prefill is finished;
                            // what follows is reasoning we never see.
                            phase = processed >= total ? .reasoning : .prefilling
                        }
                    case .chunk(let text):
                        await MainActor.run {
                            if timeToFirstToken == nil {
                                timeToFirstToken = Date().timeIntervalSince(started)
                            }
                            phase = .streaming
                            output += text
                        }
                    case .completed(let promptTime, let tokensPerSecond, let peakBytes):
                        await MainActor.run {
                            stats = String(
                                format: "%.1f tok/s · prefill %.1fs · %.2f GB peak",
                                tokensPerSecond, promptTime,
                                Double(peakBytes) / 1_073_741_824)
                        }
                    case .failed(let message):
                        await MainActor.run { errorMessage = message }
                    }
                }
            } catch {
                await MainActor.run { errorMessage = String(describing: error) }
            }

            await MainActor.run {
                phase = .idle
                prefill = nil
                isGenerating = false
            }
        }
    }
}
