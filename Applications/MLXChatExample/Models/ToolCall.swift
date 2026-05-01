//
//  ToolCall.swift
//  MLXChatExample
//

import Foundation

/// One assistant-issued tool call surfaced in the UI. `argumentsRaw`
/// accumulates as `functionCallArgumentsDelta` events arrive; `result`
/// fills in when the matching `functionCallOutput` lands.
@Observable
class ToolCall: Identifiable {
    /// Parser-minted `fc_…` id. Used for `Identifiable` and for matching
    /// `functionCallArgumentsDelta.itemId`.
    let id: String

    /// Parser-minted `call_…` id used to match the eventual
    /// `functionCallOutput` back to this call.
    let callId: String

    /// Function name the model wants to invoke.
    let name: String

    /// Raw JSON arguments string, growing one delta at a time.
    var argumentsRaw: String

    /// Tool result once dispatch returns; `nil` while in flight.
    var result: String?

    init(
        id: String, callId: String, name: String,
        argumentsRaw: String = "", result: String? = nil
    ) {
        self.id = id
        self.callId = callId
        self.name = name
        self.argumentsRaw = argumentsRaw
        self.result = result
    }
}
