import ArgumentParser
import Foundation

extension URL: @retroactive ExpressibleByArgument {
    public init?(argument: String) {
        let expanded = NSString(string: argument).expandingTildeInPath

        if argument.contains("://"), let remote = URL(string: argument), remote.scheme != nil {
            self = remote
            return
        }

        self.init(fileURLWithPath: expanded)
    }
}
