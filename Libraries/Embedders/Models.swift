// Copyright © 2024 Apple Inc.

import Foundation
import Hub

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The python tokenizers have a very rich set of implementations and configuration.  The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
public struct ModelConfiguration: Sendable {

    public enum Identifier: Sendable {
        case id(String)
        case directory(URL)
    }

    public var id: Identifier

    public var name: String {
        switch id {
        case .id(let string):
            string
        case .directory(let url):
            url.deletingLastPathComponent().lastPathComponent + "/" + url.lastPathComponent
        }
    }

    /// pull the tokenizer from an alternate id
    public let tokenizerId: String?

    /// overrides for TokenizerModel/knownTokenizers -- useful before swift-transformers is updated
    public let overrideTokenizer: String?

    public init(
        id: String, tokenizerId: String? = nil, overrideTokenizer: String? = nil
    ) {
        self.id = .id(id)
        self.tokenizerId = tokenizerId
        self.overrideTokenizer = overrideTokenizer
    }

    public init(
        directory: URL, tokenizerId: String? = nil, overrideTokenizer: String? = nil
    ) {
        self.id = .directory(directory)
        self.tokenizerId = tokenizerId
        self.overrideTokenizer = overrideTokenizer
    }

    public func modelDirectory(hub: HubApi = HubApi()) -> URL {
        switch id {
        case .id(let id):
            // download the model weights and config
            let repo = Hub.Repo(id: id)
            return hub.localRepoLocation(repo)

        case .directory(let directory):
            return directory
        }
    }

    @MainActor
    public static var registry = [String: ModelConfiguration]()

    @MainActor
    public static func register(configurations: [ModelConfiguration]) {
        bootstrap()

        for c in configurations {
            registry[c.name] = c
        }
    }

    @MainActor
    public static func configuration(id: String) -> ModelConfiguration {
        bootstrap()

        if let c = registry[id] {
            return c
        } else {
            return ModelConfiguration(id: id)
        }
    }

    @MainActor
    public static var models: some Collection<ModelConfiguration> & Sendable {
        bootstrap()
        return Self.registry.values
    }
}

extension ModelConfiguration {
    public static let bge_micro = ModelConfiguration(id: "TaylorAI/bge-micro-v2")
    public static let gte_tiny = ModelConfiguration(id: "TaylorAI/gte-tiny")
    public static let minilm_l6 = ModelConfiguration(id: "sentence-transformers/all-MiniLM-L6-v2")
    public static let snowflake_xs = ModelConfiguration(id: "Snowflake/snowflake-arctic-embed-xs")
    public static let minilm_l12 = ModelConfiguration(id: "sentence-transformers/all-MiniLM-L12-v2")
    public static let bge_small = ModelConfiguration(id: "BAAI/bge-small-en-v1.5")
    public static let multilingual_e5_small = ModelConfiguration(
        id: "intfloat/multilingual-e5-small")
    public static let bge_base = ModelConfiguration(id: "BAAI/bge-base-en-v1.5")
    public static let nomic_text_v1 = ModelConfiguration(id: "nomic-ai/nomic-embed-text-v1")
    public static let nomic_text_v1_5 = ModelConfiguration(id: "nomic-ai/nomic-embed-text-v1.5")
    public static let bge_large = ModelConfiguration(id: "BAAI/bge-large-en-v1.5")
    public static let snowflake_lg = ModelConfiguration(id: "Snowflake/snowflake-arctic-embed-l")
    public static let bge_m3 = ModelConfiguration(id: "BAAI/bge-m3")
    public static let mixedbread_large = ModelConfiguration(
        id: "mixedbread-ai/mxbai-embed-large-v1")

    private enum BootstrapState: Sendable {
        case idle
        case bootstrapping
        case bootstrapped
    }

    @MainActor
    static private var bootstrapState = BootstrapState.idle

    @MainActor
    static func bootstrap() {
        switch bootstrapState {
        case .idle:
            bootstrapState = .bootstrapping
            register(configurations: [
                bge_micro,
                gte_tiny,
                minilm_l6,
                snowflake_xs,
                minilm_l12,
                bge_small,
                multilingual_e5_small,
                bge_base,
                nomic_text_v1,
                nomic_text_v1_5,
                bge_large,
                snowflake_lg,
                bge_m3,
                mixedbread_large,
            ])
            bootstrapState = .bootstrapped

        case .bootstrapping:
            break

        case .bootstrapped:
            break
        }
    }
}
