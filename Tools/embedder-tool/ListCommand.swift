import ArgumentParser
import Foundation
import MLXEmbedders
import MLXLMCommon

struct ListCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List registered embedder model configurations"
    )

    @Flag(name: .long, help: "Include models registered from local directories")
    var includeDirectories = false

    func run() async throws {
        let models = await MainActor.run { Array(EmbedderRegistry.shared.models) }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }

        for configuration in models {
            switch configuration.id {
            case .id(let id, let revision):
                print("\(id)/\(revision)")
            case .directory(let url):
                if includeDirectories {
                    print(url.path)
                }
            }
        }
    }
}
