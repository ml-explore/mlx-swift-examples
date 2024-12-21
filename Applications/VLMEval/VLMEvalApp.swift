// Copyright © 2024 Apple Inc.

import SwiftUI

@main
struct VLMEvalApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(DeviceStat())
        }
    }
}
