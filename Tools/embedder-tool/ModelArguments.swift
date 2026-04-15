// Copyright © 2025 Apple Inc.

import ArgumentParser
import Foundation
import HuggingFace
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

struct ModelArguments: ParsableArguments {

    @Option(
        name: .long,
        help: "Name of the embedder model configuration or absolute path to a local directory.")
    var model: String?

    @Option(name: .long, help: "Directory used for downloading model assets from the Hub.")
    var download: URL?

    @MainActor
    func configuration(default defaultConfiguration: ModelConfiguration) -> ModelConfiguration {
        guard let model else {
            return defaultConfiguration
        }

        if let localConfiguration = resolveLocalModelPath(model) {
            return localConfiguration
        }

        return ModelConfiguration(id: model)
    }

    var downloadURL: URL? {
        download?.standardizedFileURL
    }
}

struct LoadedEmbedderModel {
    let configuration: ModelConfiguration
    let container: EmbedderModelContainer
}

extension ModelArguments {

    func load(default defaultConfiguration: ModelConfiguration) async throws -> LoadedEmbedderModel
    {
        let configuration = await configuration(default: defaultConfiguration)
        let hub = #hubDownloader
        let loader = #huggingFaceTokenizerLoader

        print("Loading model \(configuration.name)...")

        let container = try await EmbedderModelFactory.shared.loadContainer(
            from: hub,
            using: loader,
            configuration: configuration,
            progressHandler: { progress in
                let percentage = Int(progress.fractionCompleted * 100)
                let previousPercentage = Int((progress.fractionCompleted - 0.01) * 100)

                if percentage % 10 == 0 && percentage != previousPercentage {
                    print("Downloading model: \(percentage)%")
                }
            }
        )

        return LoadedEmbedderModel(configuration: configuration, container: container)
    }

    var downloader: any Downloader {
        let client =
            if let download {
                HubClient(cache: HubCache(cacheDirectory: download))
            } else {
                HubClient()
            }
        let downloader = #hubDownloader(client)
        return downloader
    }
}

extension ModelArguments {
    private func resolveLocalModelPath(_ value: String) -> ModelConfiguration? {
        let expanded = NSString(string: value).expandingTildeInPath
        let candidate = URL(fileURLWithPath: expanded, isDirectory: true)
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        {
            return ModelConfiguration(directory: candidate.standardizedFileURL)
        }

        if let url = URL(string: value), url.isFileURL {
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
                isDir.boolValue
            {
                return ModelConfiguration(directory: url.standardizedFileURL)
            }
        }

        return nil
    }
}
