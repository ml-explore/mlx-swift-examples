// Copyright © 2025 Apple Inc.

import SwiftUI
import MLXLMCommon
import MLXLLM
import MLX

@main
struct LLMBasicApp: App {
    
    init() {
        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
    }
    
    @State var loader = ModelLoader()
    
    var body: some Scene {
        WindowGroup {
            ContentView(loader: loader)
        }
    }
}
