// Copyright 2024 Apple Inc.

import AVKit
import AsyncAlgorithms
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM
import PhotosUI
import SwiftUI

#if os(iOS) || os(visionOS)
    typealias PlatformImage = UIImage
#else
    typealias PlatformImage = NSImage
#endif

let videoSystemPrompt =
    "Focus only on describing the key dramatic action or notable event occurring in this video segment. Skip general context or scene-setting details unless they are crucial to understanding the main action."
let imageSystemPrompt =
    "You are an image understanding model capable of describing the salient features of any image."

private let vlmDebugEnabled = ProcessInfo.processInfo.environment["QWEN3VL_DEBUG"] != nil

private func debugLog(_ message: @autoclosure () -> String) {
    guard vlmDebugEnabled else { return }
    print("[VLMEval] \(message())")
}

struct ContentView: View {

    @State var llm = VLMEvaluator()
    @Environment(DeviceStat.self) private var deviceStat

    @State private var selectedImage: PlatformImage? = nil {
        didSet {
            if selectedImage != nil {
                selectedVideoURL = nil
                player = nil
            }
        }
    }
    @State private var selectedVideoURL: URL? {
        didSet {
            if let selectedVideoURL {
                player = AVPlayer(url: selectedVideoURL)
                selectedImage = nil
            }
        }
    }
    @State private var showingImagePicker = false
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var player: AVPlayer? = nil

    private var currentImageURL: URL? {
        selectedImage == nil && selectedVideoURL == nil
            ? URL(
                string:
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
            ) : nil
    }

    var body: some View {
        VStack(alignment: .leading) {
            VStack {
                HStack {
                    Text(llm.modelInfo)
                        .textFieldStyle(.roundedBorder)

                    Spacer()

                    Text(llm.stat)
                }

                VStack {
                    if let player {
                        VideoPlayer(player: player)
                            .frame(height: 300)
                            .cornerRadius(12)
                    } else if let selectedImage {
                        Group {
                            #if os(iOS) || os(visionOS)
                                Image(uiImage: selectedImage)
                                    .resizable()
                            #else
                                Image(nsImage: selectedImage)
                                    .resizable()
                            #endif
                        }
                        .scaledToFit()
                        .cornerRadius(12)
                        .frame(height: 300)
                    } else if let imageURL = currentImageURL {
                        AsyncImage(url: imageURL) { phase in
                            switch phase {
                            case .empty:
                                ProgressView()
                            case .success(let image):
                                image
                                    .resizable()
                                    .scaledToFit()
                                    .cornerRadius(12)
                                    .frame(height: 200)
                            case .failure:
                                Image(systemName: "photo.badge.exclamationmark")
                            @unknown default:
                                EmptyView()
                            }
                        }
                    }

                    HStack {
                        #if os(iOS) || os(visionOS)
                            PhotosPicker(
                                selection: $selectedItem,
                                matching: PHPickerFilter.any(of: [
                                    PHPickerFilter.images, PHPickerFilter.videos,
                                ])
                            ) {
                                Label("Select Image/Video", systemImage: "photo.badge.plus")
                            }
                            .onChange(of: selectedItem) {
                                Task {
                                    debugLog("PhotosPicker selection updated")
                                    if let video = try? await selectedItem?.loadTransferable(
                                        type: TransferableVideo.self)
                                    {
                                        debugLog("PhotosPicker -> loaded video transferable")
                                        selectedVideoURL = video.url
                                    } else if let data = try? await selectedItem?.loadTransferable(
                                        type: Data.self)
                                    {
                                        debugLog("PhotosPicker -> loaded image data")
                                        selectedImage = PlatformImage(data: data)
                                    }
                                }
                            }
                        #else
                            Button("Select Image/Video") {
                                showingImagePicker = true
                            }
                            .fileImporter(
                                isPresented: $showingImagePicker,
                                allowedContentTypes: [.image, .movie]
                            ) { result in
                                switch result {
                                case .success(let file):
                                    debugLog("File importer success -> \(file.path())")
                                    Task { @MainActor in
                                        do {
                                            let data = try loadData(from: file)
                                            if let image = PlatformImage(data: data) {
                                                debugLog("File importer -> decoded image")
                                                selectedImage = image
                                            } else if let fileType = UTType(
                                                filenameExtension: file.pathExtension),
                                                fileType.conforms(to: .movie)
                                            {
                                                if let sandboxURL = try? loadVideoToSandbox(
                                                    from: file)
                                                {
                                                    debugLog("File importer -> copied video to sandbox")
                                                    selectedVideoURL = sandboxURL
                                                }
                                            } else {
                                                print("Failed to create image from data")
                                            }
                                        } catch {
                                            print(
                                                "Failed to load image: \(error.localizedDescription)"
                                            )
                                        }
                                    }
                                case .failure(let error):
                                    debugLog("File importer error -> \(error)")
                                }
                            }
                        #endif

                        if selectedImage != nil {
                            Button("Clear", role: .destructive) {
                                selectedImage = nil
                                selectedItem = nil
                            }
                        }
                    }
                }
                .padding()

                HStack {
                    Spacer()
                    if llm.running {
                        ProgressView()
                            .frame(maxHeight: 20)
                        Spacer()
                    }
                }
            }

            ScrollView(.vertical) {
                ScrollViewReader { sp in
                    Text(llm.output)
                        .textSelection(.enabled)
                        .onChange(of: llm.output) { _, _ in
                            sp.scrollTo("bottom")
                        }

                    Spacer()
                        .frame(width: 1, height: 1)
                        .id("bottom")
                }
            }
            .frame(minHeight: 200)

            HStack {
                TextField("prompt", text: Bindable(llm).prompt)
                    .onSubmit(generate)
                    .disabled(llm.running)
                    #if os(visionOS)
                        .textFieldStyle(.roundedBorder)
                    #endif
                Button(llm.running ? "stop" : "generate", action: llm.running ? cancel : generate)
            }
        }
        .onAppear {
            selectedVideoURL = URL(
                string:
                    "https://videos.pexels.com/video-files/4066325/4066325-uhd_2560_1440_24fps.mp4")!
        }
        #if os(visionOS)
            .padding(40)
        #else
            .padding()
        #endif
        .toolbar {
            ToolbarItem {
                Label(
                    "Memory Usage: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))",
                    systemImage: "info.circle.fill"
                )
                .labelStyle(.titleAndIcon)
                .padding(.horizontal)
                .help(
                    Text(
                        """
                        Active Memory: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))/\(GPU.memoryLimit.formatted(.byteCount(style: .memory)))
                        Cache Memory: \(deviceStat.gpuUsage.cacheMemory.formatted(.byteCount(style: .memory)))/\(GPU.cacheLimit.formatted(.byteCount(style: .memory)))
                        Peak Memory: \(deviceStat.gpuUsage.peakMemory.formatted(.byteCount(style: .memory)))
                        """
                    )
                )
            }
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
            debugLog("ContentView.task -> triggering load")
            do {
                _ = try await llm.load()
                debugLog("ContentView.task -> load succeeded")
            } catch {
                debugLog("ContentView.task -> load failed \(error)")
            }
        }
    }

    private func generate() {
        Task {
            if let selectedImage = selectedImage {
                #if os(iOS) || os(visionOS)
                    let ciImage = CIImage(image: selectedImage)
                    debugLog("generate() action -> using selected UIImage/UIImage")
                    llm.generate(image: ciImage ?? CIImage(), videoURL: nil)
                #else
                    if let cgImage = selectedImage.cgImage(
                        forProposedRect: nil, context: nil, hints: nil)
                    {
                        let ciImage = CIImage(cgImage: cgImage, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
                        debugLog("generate() action -> using converted CGImage input")
                        llm.generate(image: ciImage, videoURL: nil)
                    }
                #endif
            } else if let imageURL = currentImageURL {
                do {
                    let (data, _) = try await URLSession.shared.data(from: imageURL)
                    if let ciImage = CIImage(data: data, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!]) {
                        debugLog("generate() action -> using default remote image")
                        llm.generate(image: ciImage, videoURL: nil)
                    }
                } catch {
                    print("Failed to load image: \(error.localizedDescription)")
                }
            } else {
                if let videoURL = selectedVideoURL {
                    debugLog("generate() action -> using selected video")
                    llm.generate(image: nil, videoURL: videoURL)
                } else {
                    debugLog("generate() action -> text-only mode (no image/video)")
                    llm.generate(image: nil, videoURL: nil)
                }
            }
        }
    }

    private func cancel() {
        llm.cancelGeneration()
    }

    #if os(macOS)
        private func loadData(from url: URL) throws -> Data {
            guard url.startAccessingSecurityScopedResource() else {
                throw NSError(
                    domain: "FileAccess", code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to access the file."])
            }
            defer { url.stopAccessingSecurityScopedResource() }
            return try Data(contentsOf: url)
        }

        private func loadVideoToSandbox(from url: URL) throws -> URL {
            guard url.startAccessingSecurityScopedResource() else {
                throw NSError(
                    domain: "FileAccess", code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to access the file."])
            }
            defer { url.stopAccessingSecurityScopedResource() }
            let sandboxURL = try SandboxFileTransfer.transferFileToTemp(from: url)
            return sandboxURL
        }
    #endif

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
@MainActor
class VLMEvaluator {

    var running = false

    var prompt = ""
    var output = ""
    var modelInfo = ""
    var stat = ""

    /// This controls which model loads. Testing Qwen3-VL-4B-8bit after fixing deepstack bug.
    let modelConfiguration = VLMRegistry.qwen3VL4BInstruct4Bit

    /// parameters controlling the output – use values appropriate for the model selected above
    /// Using Qwen3-VL recommended settings: temp=0.7, top_p=0.8, repetition_penalty=1.2
    let generateParameters = MLXLMCommon.GenerateParameters(
        maxTokens: 800, temperature: 0.7, topP: 0.8, repetitionPenalty: 1.2)
    let updateInterval = Duration.seconds(0.25)

    /// A task responsible for handling the generation process.
    var generationTask: Task<Void, Error>?

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    var loadState = LoadState.idle

    /// load and return the model -- can be called multiple times, subsequent calls will
    /// just return the loaded model
    func load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            // limit the buffer cache
            debugLog("load() idle -> configuring GPU cache and starting load for \(modelConfiguration.id)")
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let modelContainer = try await VLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) { [modelConfiguration] progress in
                debugLog(
                    "load() progress -> \(Int(progress.fractionCompleted * 100))% for \(modelConfiguration.id)")
                Task { @MainActor in
                    self.modelInfo =
                        "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }

            let numParams = await modelContainer.perform { context in
                debugLog("load() -> querying parameter count")
                return context.model.numParameters()
            }

            self.prompt = modelConfiguration.defaultPrompt
            self.modelInfo = "Loaded \(modelConfiguration.id). Weights: \(numParams / (1024*1024))M"
            debugLog("load() complete -> params=\(numParams)")
            loadState = .loaded(modelContainer)
            return modelContainer

        case .loaded(let modelContainer):
            debugLog("load() reuse existing container")
            return modelContainer
        }
    }

    private func generate(prompt: String, image: CIImage?, videoURL: URL?) async {

        self.output = ""
        debugLog(
            "generate(prompt:image:video:) start -> promptCount=\(prompt.count) imageSet=\(image != nil) videoSet=\(videoURL != nil)")

        do {
            let modelContainer = try await load()
            debugLog("generate -> container ready")

            // each time you generate you will get something new
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))
            debugLog("generate -> RNG seeded")

            try await modelContainer.perform { (context: ModelContext) -> Void in
                let images: [UserInput.Image] = if let image { [.ciImage(image)] } else { [] }
                let videos: [UserInput.Video] = if let videoURL { [.url(videoURL)] } else { [] }

                debugLog("model.perform -> images=\(images.count) videos=\(videos.count) promptCount=\(prompt.count)")

                let systemPrompt =
                    if !videos.isEmpty {
                        videoSystemPrompt
                    } else if !images.isEmpty {
                        imageSystemPrompt
                    } else { "You are a helpful assistant." }

                let chat: [Chat.Message] = [
                    .system(systemPrompt),
                    .user(prompt, images: images, videos: videos),
                ]
                
                debugLog("Chat messages: system='\(systemPrompt)' user='\(prompt)'")

                var userInput = UserInput(chat: chat)
                userInput.processing.resize = .init(width: 448, height: 448)

                let lmInput = try await context.processor.prepare(input: userInput)

                let stream = try MLXLMCommon.generate(
                    input: lmInput, parameters: generateParameters, context: context)

                debugLog("Starting token stream...")
                var tokenCount = 0
                // generate and output in batches
                for await batch in stream._throttle(
                    for: updateInterval, reducing: Generation.collect)
                {
                    tokenCount += batch.count
                    let output = batch.compactMap { $0.chunk }.joined(separator: "")
                    if !output.isEmpty {
                        debugLog("Received output chunk: '\(output)'")
                        Task { @MainActor [output] in
                            self.output += output
                        }
                    }

                    if let completion = batch.compactMap({ $0.info }).first {
                        Task { @MainActor in
                            self.stat = "\(completion.tokensPerSecond) tokens/s"
                        }
                    }
                }
                debugLog("model.perform -> streaming completed, total tokens: \(tokenCount)")
            }
        } catch {
            output = "Failed: \(error)"
            debugLog("generate -> error \(error)")
        }
    }

    func generate(image: CIImage?, videoURL: URL?) {
        guard !running else { return }
        debugLog("public generate() -> image?=\(image != nil) video?=\(videoURL != nil)")
        let currentPrompt = prompt
        prompt = ""
        generationTask = Task {
            running = true
            debugLog("generationTask -> started")
            await generate(prompt: currentPrompt, image: image, videoURL: videoURL)
            running = false
            debugLog("generationTask -> finished")
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        running = false
        debugLog("cancelGeneration -> cancelled")
    }
}

#if os(iOS) || os(visionOS)
    struct TransferableVideo: Transferable {
        let url: URL

        static var transferRepresentation: some TransferRepresentation {
            FileRepresentation(contentType: .movie) { movie in
                SentTransferredFile(movie.url)
            } importing: { received in
                let sandboxURL = try SandboxFileTransfer.transferFileToTemp(from: received.file)
                return .init(url: sandboxURL)
            }
        }
    }
#endif

struct SandboxFileTransfer {
    static func transferFileToTemp(from sourceURL: URL) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let sandboxURL = tempDir.appendingPathComponent(sourceURL.lastPathComponent)

        if FileManager.default.fileExists(atPath: sandboxURL.path()) {
            try FileManager.default.removeItem(at: sandboxURL)
        }

        try FileManager.default.copyItem(at: sourceURL, to: sandboxURL)
        return sandboxURL
    }
}
