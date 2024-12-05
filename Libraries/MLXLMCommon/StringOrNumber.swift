// Copyright © 2024 Apple Inc.

import Foundation

/// Representation of a heterogenous type in a JSON configuration file.
///
/// This can be: a string, a numeric value or an array of numeric values.
/// There are methods to do unwrapping, see e.g. ``asFloat()`` and
/// ``asFloats()`` or callers can switch on the enum.
public enum StringOrNumber: Codable, Equatable, Sendable {
    case string(String)
    case int(Int)
    case float(Float)
    case ints([Int])
    case floats([Float])

    public init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer()

        if let v = try? values.decode(Int.self) {
            self = .int(v)
        } else if let v = try? values.decode(Float.self) {
            self = .float(v)
        } else if let v = try? values.decode([Int].self) {
            self = .ints(v)
        } else if let v = try? values.decode([Float].self) {
            self = .floats(v)
        } else {
            let v = try values.decode(String.self)
            self = .string(v)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .int(let v): try container.encode(v)
        case .float(let v): try container.encode(v)
        case .ints(let v): try container.encode(v)
        case .floats(let v): try container.encode(v)
        }
    }

    /// Return the value as an optional array of integers.
    ///
    /// This will not coerce `Float` or `String` to `Int`.
    public func asInts() -> [Int]? {
        switch self {
        case .string(let string): nil
        case .int(let v): [v]
        case .float(let float): nil
        case .ints(let array): array
        case .floats(let array): nil
        }
    }

    /// Return the value as an optional integer.
    ///
    /// This will not coerce `Float` or `String` to `Int`.
    public func asInt() -> Int? {
        switch self {
        case .string(let string): nil
        case .int(let v): v
        case .float(let float): nil
        case .ints(let array): array.count == 1 ? array[0] : nil
        case .floats(let array): nil
        }
    }

    /// Return the value as an optional array of floats.
    ///
    /// This will not coerce `Int` or `String` to `Float`.
    public func asFloats() -> [Float]? {
        switch self {
        case .string(let string): nil
        case .int(let v): [Float(v)]
        case .float(let float): [float]
        case .ints(let array): array.map { Float($0) }
        case .floats(let array): array
        }
    }

    /// Return the value as an optional float.
    ///
    /// This will not coerce `Int` or `String` to `Float`.
    public func asFloat() -> Float? {
        switch self {
        case .string(let string): nil
        case .int(let v): Float(v)
        case .float(let float): float
        case .ints(let array): array.count == 1 ? Float(array[0]) : nil
        case .floats(let array): array.count == 1 ? array[0] : nil
        }
    }
}
