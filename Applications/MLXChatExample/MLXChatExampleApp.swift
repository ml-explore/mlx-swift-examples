//
//  MLXChatExampleApp.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import SwiftUI

@main
struct MLXChatExampleApp: App {
    var body: some Scene {
        WindowGroup {
            ChatView(viewModel: ChatViewModel(mlxService: MLXService()))
        }
    }
}
