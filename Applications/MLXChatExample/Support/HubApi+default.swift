//
//  HubApi+default.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 21.04.2025.
//

import Foundation
@preconcurrency import Hub

/// Extension providing a default HubApi instance for downloading model files
extension HubApi {
    /// Default HubApi instance configured to download models to the user's Downloads directory
    /// under a 'huggingface' subdirectory.
    #if os(macOS)
        static let `default` = HubApi(
            downloadBase: URL.downloadsDirectory.appending(path: "huggingface")
        )
    #else
        static let `default` = HubApi(
            downloadBase: URL.cachesDirectory.appending(path: "huggingface")
        )
    #endif
}
