import Foundation

/// `swift-transformers` also declares a public `Decoder` and it conflicts with the `Codable`
/// implementations.
public typealias Decoder = Swift.Decoder
