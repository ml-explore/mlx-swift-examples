// Copyright © 2025 Apple Inc.

import Foundation

/// Utility functions for formatting values
enum FormatUtilities {

    /// Formats a byte count into a human-readable string with appropriate units
    /// - Parameter bytes: The number of bytes to format
    /// - Returns: A formatted string (e.g., "2.5 GB", "128 MB", "512 KB")
    static func formatMemory(_ bytes: Int) -> String {
        let kb = Double(bytes) / 1024
        let mb = kb / 1024
        let gb = mb / 1024

        if gb >= 1 {
            return String(format: "%.2f GB", gb)
        } else if mb >= 1 {
            return String(format: "%.0f MB", mb)
        } else if kb >= 1 {
            return String(format: "%.0f KB", kb)
        } else {
            return "0 KB"
        }
    }
}
