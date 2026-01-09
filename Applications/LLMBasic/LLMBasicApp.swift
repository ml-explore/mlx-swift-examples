// Copyright © 2025 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon
import SwiftUI

@main
struct LLMBasicApp: App {

    init() {
        Memory.cacheLimit = 20 * 1024 * 1024
    }

    @State var loader = ModelLoader()

    var body: some Scene {
        WindowGroup {
            ContentView(loader: loader)
        }
    }
}
