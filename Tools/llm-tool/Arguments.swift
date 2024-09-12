// Copyright © 2024 Apple Inc.

import ArgumentParser
import Foundation

/// Extension to allow URL command line arguments.
#if swift(>=5.10)
    extension URL: @retroactive ExpressibleByArgument {
        public init?(argument: String) {
            if argument.contains("://") {
                self.init(string: argument)
            } else {
                self.init(filePath: argument)
            }
        }
    }
#else
    extension URL: ExpressibleByArgument {
        public init?(argument: String) {
            if argument.contains("://") {
                self.init(string: argument)
            } else {
                self.init(filePath: argument)
            }
        }
    }
#endif
