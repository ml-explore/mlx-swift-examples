import Foundation

enum DiagnosticKind {
    case info
    case warning
    case error

    fileprivate var prefix: String {
        switch self {
        case .info:
            return ""
        case .warning:
            return "warning: "
        case .error:
            return "error: "
        }
    }
}

func writeDiagnostic(_ message: String, kind: DiagnosticKind = .info) {
    guard let data = (kind.prefix + message + "\n").data(using: .utf8) else {
        return
    }
    FileHandle.standardError.write(data)
}
