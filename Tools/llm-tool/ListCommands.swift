// Copyright © 2025 Apple Inc.

import ArgumentParser
import Foundation
import MLXLLM
import MLXVLM

struct ListCommands: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "list registered model configurations",
        subcommands: [
            ListLLMCommand.self, ListVLMCommand.self,
        ]
    )
}

struct ListLLMCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "llms",
        abstract: "List registered LLM model configurations"
    )

    func run() async throws {
        for configuration in LLMRegistry.shared.models {
            switch configuration.id {
            case .id(let id): print(id)
            case .directory: break
            }
        }
    }
}

struct ListVLMCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "vlms",
        abstract: "List registered VLM model configurations"
    )

    func run() async throws {
        for configuration in VLMRegistry.shared.models {
            switch configuration.id {
            case .id(let id): print(id)
            case .directory: break
            }
        }
    }
}
