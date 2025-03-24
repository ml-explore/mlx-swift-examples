// Copyright © 2024 Apple Inc.

import MLX
import StableDiffusion
import SwiftUI

struct ContentView: View {

    @State var prompt = "dismal swamp, dense, very dark, realistic"
    @State var negativePrompt = ""
    @State var evaluator = StableDiffusionEvaluator()
    @State var showProgress = false

    var body: some View {
        VStack {
            HStack {
                if let progress = evaluator.progress {
                    ProgressView(progress.title, value: progress.current, total: progress.limit)
                }
            }
            .frame(height: 20)

            Spacer()
            if let image = evaluator.image {
                Image(image, scale: 1.0, label: Text(""))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(minHeight: 200)
            }
            Spacer()

            Grid {
                GridRow {
                    TextField("prompt", text: $prompt)
                        .onSubmit(generate)
                        .disabled(evaluator.progress != nil)
                        #if os(visionOS)
                            .textFieldStyle(.roundedBorder)
                        #endif

                    Button(action: { prompt = "" }) {
                        Label("clear", systemImage: "xmark.circle.fill").font(.system(size: 10))
                    }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.plain)

                    Button("generate", action: generate)
                        .disabled(evaluator.progress != nil)
                        .keyboardShortcut("r")
                }
                if evaluator.modelFactory.canShowProgress
                    || evaluator.modelFactory.canUseNegativeText
                {
                    GridRow {
                        if evaluator.modelFactory.canUseNegativeText {
                            TextField("negative prompt", text: $negativePrompt)
                                .onSubmit(generate)
                                .disabled(evaluator.progress != nil)
                                #if os(visionOS)
                                    .textFieldStyle(.roundedBorder)
                                #endif
                            Button(action: { prompt = "" }) {
                                Label("clear", systemImage: "xmark.circle.fill").font(
                                    .system(size: 10))
                            }
                            .labelStyle(.iconOnly)
                            .buttonStyle(.plain)
                        } else {
                            EmptyView()
                            EmptyView()
                        }

                        if evaluator.modelFactory.canShowProgress {
                            Toggle("Show Progress", isOn: $showProgress)
                        }
                    }
                }
            }
            .frame(minWidth: 300)
        }
        .padding()
    }

    private func generate() {
        Task {
            await evaluator.generate(
                prompt: prompt, negativePrompt: negativePrompt, showProgress: showProgress)
        }
    }
}

/// Progress reporting with a title.
struct Progress: Equatable {
    let title: String
    let current: Double
    let limit: Double
}

/// Async model factory
actor ModelFactory {

    enum LoadState {
        case idle
        case loading(Task<ModelContainer<TextToImageGenerator>, Error>)
        case loaded(ModelContainer<TextToImageGenerator>)
    }

    enum SDError: LocalizedError {
        case unableToLoad

        var errorDescription: String? {
            switch self {
            case .unableToLoad:
                return String(
                    localized:
                        "Unable to load the Stable Diffusion model. Please check your internet connection or available storage space."
                )
            }
        }
    }

    public nonisolated let configuration = StableDiffusionConfiguration.presetSDXLTurbo

    /// if true we show UI that lets users see the intermediate steps
    public nonisolated let canShowProgress: Bool

    /// if true we show UI to give negative text
    public nonisolated let canUseNegativeText: Bool

    private var loadState = LoadState.idle
    private var loadConfiguration = LoadConfiguration(float16: true, quantize: false)

    public nonisolated let conserveMemory: Bool

    init() {
        let defaultParameters = configuration.defaultParameters()
        self.canShowProgress = defaultParameters.steps > 4
        self.canUseNegativeText = defaultParameters.cfgWeight > 1

        // this will be true e.g. if the computer has 8G of memory or less
        self.conserveMemory = MLX.GPU.memoryLimit < 8 * 1024 * 1024 * 1024

        if conserveMemory {
            print("conserving memory")
            loadConfiguration.quantize = true
            MLX.GPU.set(cacheLimit: 1 * 1024 * 1024)
            MLX.GPU.set(memoryLimit: 3 * 1024 * 1024 * 1024)
        } else {
            MLX.GPU.set(cacheLimit: 256 * 1024 * 1024)
        }
    }

    public func load(reportProgress: @escaping @Sendable (Progress) -> Void) async throws
        -> ModelContainer<TextToImageGenerator>
    {
        switch loadState {
        case .idle:
            let task = Task {
                do {
                    try await configuration.download { progress in
                        if progress.fractionCompleted < 0.99 {
                            reportProgress(
                                .init(
                                    title: "Download", current: progress.fractionCompleted * 100,
                                    limit: 100))
                        }
                    }
                } catch {
                    let nserror = error as NSError
                    if nserror.domain == NSURLErrorDomain
                        && nserror.code == NSURLErrorNotConnectedToInternet
                    {
                        // Internet connection appears to be offline -- fall back to loading from
                        // the local directory
                        reportProgress(.init(title: "Offline", current: 100, limit: 100))
                    } else {
                        throw error
                    }
                }

                let container = try ModelContainer<TextToImageGenerator>.createTextToImageGenerator(
                    configuration: configuration, loadConfiguration: loadConfiguration)

                await container.setConserveMemory(conserveMemory)

                try await container.perform { model in
                    reportProgress(.init(title: "Loading weights", current: 0, limit: 1))
                    if !conserveMemory {
                        model.ensureLoaded()
                    }
                }

                return container
            }
            self.loadState = .loading(task)

            let container = try await task.value

            if conserveMemory {
                // if conserving memory return the model but do not keep it in memory
                self.loadState = .idle
            } else {
                // cache the model in memory to make it faster to run with new prompts
                self.loadState = .loaded(container)
            }

            return container

        case .loading(let task):
            let generator = try await task.value
            return generator

        case .loaded(let generator):
            return generator
        }
    }

}

@Observable @MainActor
class StableDiffusionEvaluator {

    var progress: Progress?
    var message: String?
    var image: CGImage?

    let modelFactory = ModelFactory()

    @Sendable
    nonisolated private func updateProgress(progress: Progress?) {
        Task { @MainActor in
            self.progress = progress
        }
    }

    @Sendable
    nonisolated private func updateImage(image: CGImage?) {
        Task { @MainActor in
            self.image = image
        }
    }

    nonisolated private func display(decoded: MLXArray) {
        let raster = (decoded * 255).asType(.uint8).squeezed()
        let image = Image(raster).asCGImage()

        Task { @MainActor in
            updateImage(image: image)
        }
    }

    func generate(prompt: String, negativePrompt: String, showProgress: Bool) async {
        progress = .init(title: "Preparing", current: 0, limit: 1)
        message = nil

        // the parameters that control the generation of the image.  See
        // EvaluateParameters for more information.  For example adjusting
        // the latentSize parameter will change the size of the generated
        // image.  imageCount could be used to generate a gallery of
        // images at the same time.
        let parameters = {
            var p = modelFactory.configuration.defaultParameters()
            p.prompt = prompt
            p.negativePrompt = negativePrompt

            // per measurement each step consumes memory that we want to conserve.  trade
            // off steps (quality) for memory
            if modelFactory.conserveMemory {
                p.steps = 1
            }

            return p
        }()

        do {
            // note: the optionals are used to discard parts of the model
            // as it runs -- this is used to conserveMemory in devices
            // with less memory
            let container = try await modelFactory.load(reportProgress: updateProgress)

            try await container.performTwoStage { generator in
                // the parameters that control the generation of the image.  See
                // EvaluateParameters for more information.  For example adjusting
                // the latentSize parameter will change the size of the generated
                // image.  imageCount could be used to generate a gallery of
                // images at the same time.
                var parameters = modelFactory.configuration.defaultParameters()
                parameters.prompt = prompt
                parameters.negativePrompt = negativePrompt

                // per measurement each step consumes memory that we want to conserve.  trade
                // off steps (quality) for memory
                if modelFactory.conserveMemory {
                    parameters.steps = 1
                }

                // generate the latent images -- this is fast as it is just generating
                // the graphs that will be evaluated below
                let latents: DenoiseIterator? = generator.generateLatents(parameters: parameters)

                // when conserveMemory is true this will discard the first part of
                // the model and just evaluate the decode portion
                return (generator.detachedDecoder(), latents)

            } second: { decoder, latents in
                var lastXt: MLXArray?
                for (i, xt) in latents!.enumerated() {
                    lastXt = nil
                    eval(xt)
                    lastXt = xt

                    if showProgress, i % 10 == 0 {
                        display(decoded: decoder(xt))
                    }

                    updateProgress(
                        progress: .init(
                            title: "Generate Latents", current: Double(i),
                            limit: Double(parameters.steps)))
                }

                if let lastXt {
                    display(decoded: decoder(lastXt))
                }
                updateProgress(progress: nil)
            }

        } catch {
            progress = nil
            message = "Failed: \(error)"
        }
    }
}
