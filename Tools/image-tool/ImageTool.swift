// Copyright © 2024 Apple Inc.

import ArgumentParser
import Foundation
import MLX
import Progress
import StableDiffusion

@main
struct ImageTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Command line tool for working with images and MLX",
        subcommands: [StableDiffusionTool.self])
}

struct StableDiffusionTool: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "sd",
        abstract: "Stable diffusion related commands",
        subcommands: [TextToImageCommand.self, ImageToImageCommand.self]
    )
}

#if swift(>=5.10)
    extension StableDiffusionConfiguration.Preset: @retroactive ExpressibleByArgument {}
#else
    extension StableDiffusionConfiguration.Preset: ExpressibleByArgument {}
#endif

struct ModelArguments: ParsableArguments, Sendable {

    @Option(name: .long, help: "stable diffusion model")
    var model: StableDiffusionConfiguration.Preset = .sdxlTurbo

    @Flag(name: .long, inversion: .prefixedNo, help: "Disable float16 conversion")
    var float16 = true

    @Flag(name: .long, help: "Enable quantization")
    var quantize = false

    var loadConfiguration: LoadConfiguration {
        LoadConfiguration(float16: float16, quantize: quantize)
    }

    func download() async throws -> StableDiffusionConfiguration {
        let configuration = model.configuration

        var progressBar: ProgressBar?
        try await configuration.download { progress in
            if progressBar == nil {
                let complete = progress.fractionCompleted
                if complete < 0.99 {
                    progressBar = ProgressBar(count: 1000)
                    if complete > 0 {
                        print("Resuming download (\(Int(complete * 100))% complete)")
                    } else {
                        print("Downloading")
                    }
                    print()
                }
            }

            let complete = Int(progress.fractionCompleted * 1000)
            progressBar?.setValue(complete)
        }

        return configuration
    }
}

/// Command line arguments for controlling generation of images
struct GenerateArguments: ParsableArguments, Sendable {

    @Option(name: .shortAndLong, help: "The message to be processed by the model")
    var prompt = "purple cow on the moon"

    @Option(name: .shortAndLong, help: "Negative prompt (requires cfg to be > 1)")
    var negativePrompt = ""

    @Option(name: .long, help: "cfg weight")
    var cfg: Float?

    @Option(name: .long, help: "number of images")
    var imageCount: Int = 1

    @Option(name: .long, help: "decoding batch size")
    var batchSize: Int = 1

    @Option(name: .long, help: "latent width (output size is 8x this value)")
    var latentWidth: Int = 64

    @Option(name: .long, help: "latent height (output size is 8x this value)")
    var latentHeight: Int = 64

    @Option(name: .long, help: "number of rows of images in the output")
    var rows: Int = 1

    @Option(name: .long, help: "number of steps")
    var steps: Int?

    @Option(name: .long, help: "The PRNG seed")
    var seed: UInt64?

    func evaluateParameters(configuration: StableDiffusionConfiguration) -> EvaluateParameters {
        var parameters = configuration.defaultParameters()
        parameters.prompt = prompt
        parameters.negativePrompt = negativePrompt
        if let cfg {
            parameters.cfgWeight = cfg
        }
        parameters.imageCount = imageCount
        parameters.decodingBatchSize = batchSize
        parameters.latentSize = [latentHeight, latentWidth]
        if let steps {
            parameters.steps = steps
        }
        if let seed {
            parameters.seed = seed
        }
        print("using seed: \(parameters.seed)")
        return parameters
    }

}

func makeGrid(images: [MLXArray], rows: Int) -> MLXArray {
    var x = concatenated(images, axis: 0)
    x = padded(x, widths: [[0, 0], [8, 8], [8, 8], [0, 0]])
    let (B, H, W, C) = x.shape4
    x = x.reshaped(rows, B / rows, H, W, C).transposed(0, 2, 1, 3, 4)
    x = x.reshaped(rows * H, B / rows * W, C)
    x = (x * 255).asType(.uint8)
    return x
}

struct TextToImageCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "text",
        abstract: "Text to image command"
    )

    @Option(name: .long, help: "output image")
    var output = URL(filePath: "/tmp/out.png")

    @OptionGroup var model: ModelArguments
    @OptionGroup var memory: MemoryArguments
    @OptionGroup var generate: GenerateArguments

    mutating func generateLatents(configuration: StableDiffusionConfiguration) throws -> (
        EvaluateParameters, ImageDecoder, MLXArray
    ) {
        // download and prepare the model
        guard
            let generator = try configuration.textToImageGenerator(
                configuration: model.loadConfiguration)
        else {
            fatalError("Unable to produce TextToImageGenerator from \(configuration.id)")
        }

        generator.ensureLoaded()
        memory.start()

        // generate the latents -- these are the iterations for generating
        // the output image.  this is just generating the evaluation graph
        let parameters = generate.evaluateParameters(configuration: configuration)
        let latents = generator.generateLatents(parameters: parameters)

        // evaluate the latents (evalue the graph) and keep the last value generated
        var lastXt: MLXArray!
        for xt in Progress(latents) {
            eval(xt)
            lastXt = xt
        }

        return (parameters, generator.detachedDecoder(), lastXt)
    }

    @MainActor
    mutating func run() async throws {
        let configuration = try await model.download()

        let (parameters, decoder, xt) = try generateLatents(configuration: configuration)

        var decoded = [MLXArray]()
        for i in Progress(
            stride(from: 0, to: parameters.imageCount, by: parameters.decodingBatchSize))
        {
            let image = decoder(xt[i ..< i + parameters.decodingBatchSize])
            eval(image)
            decoded.append(image)
        }

        let grid = makeGrid(images: decoded, rows: generate.rows)
        try Image(grid).save(url: output)
    }
}

struct ImageToImageCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "image",
        abstract: "Image to image command"
    )

    @Option(name: .long, help: "input image")
    var input: URL

    @Option(name: .long, help: "maximum edge of the input image -- scale to fit this size")
    var maxEdge: Int = 1024

    @Option(name: .long, help: "output image")
    var output = URL(filePath: "/tmp/out.png")

    @Option(name: .long, help: "noise strength")
    var strength: Float = 0.9

    @OptionGroup var model: ModelArguments
    @OptionGroup var memory: MemoryArguments
    @OptionGroup var generate: GenerateArguments

    mutating func generateLatents(configuration: StableDiffusionConfiguration) throws -> (
        EvaluateParameters, ImageDecoder, MLXArray
    ) {

        let image = try Image(url: self.input, maximumEdge: maxEdge)
        let input = (image.data.asType(.float32) / 255) * 2 - 1

        guard
            let generator = try configuration.imageToImageGenerator(
                configuration: model.loadConfiguration)
        else {
            fatalError("Unable to produce TextToImageGenerator from \(configuration.id)")
        }

        generator.ensureLoaded()
        memory.start()

        // adjust the steps based on the strength
        if Int(Float(generate.evaluateParameters(configuration: configuration).steps) * strength)
            < 1
        {
            generate.steps = Int(ceil(1 / strength))
        }

        let parameters = generate.evaluateParameters(configuration: configuration)
        let latents = generator.generateLatents(
            image: input, parameters: parameters, strength: strength)

        var lastXt: MLXArray!
        for xt in Progress(latents) {
            eval(xt)
            lastXt = xt
        }

        return (parameters, generator.detachedDecoder(), lastXt)
    }

    @MainActor
    mutating func run() async throws {
        let configuration = try await model.download()

        let (parameters, decoder, xt) = try generateLatents(configuration: configuration)

        var decoded = [MLXArray]()
        for i in Progress(
            stride(from: 0, to: parameters.imageCount, by: parameters.decodingBatchSize))
        {
            let image = decoder(xt[i ..< i + parameters.decodingBatchSize])
            eval(image)
            decoded.append(image)
        }

        let grid = makeGrid(images: decoded, rows: generate.rows)
        try Image(grid).save(url: output)
    }
}
