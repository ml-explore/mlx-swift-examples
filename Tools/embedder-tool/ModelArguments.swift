// Copyright © 2025 Apple Inc.

import ArgumentParser
import Foundation
import MLXEmbedders

struct ModelArguments: ParsableArguments {

    @Option(name: .long, help: "Name of the embedder model configuration or absolute path to a local directory.")
    var model: String?

    @Option(name: .long, help: "Directory used for downloading model assets from the Hub.")
    var download: String?

    @MainActor
    func configuration(default defaultConfiguration: ModelConfiguration) -> ModelConfiguration {
        guard let model else {
            return defaultConfiguration
        }

        if model.hasPrefix("/") {
            return ModelConfiguration(directory: URL(filePath: model))
        }

        return ModelConfiguration.configuration(id: model)
    }

    var downloadURL: URL? {
        download.map { URL(fileURLWithPath: $0) }
    }
}
