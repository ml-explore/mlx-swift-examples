//
//  GenerationInfoView.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 21.04.2025.
//

import SwiftUI

struct GenerationInfoView: View {
    let tokensPerSecond: Double

    var body: some View {
        Text("\(tokensPerSecond, format: .number.precision(.fractionLength(2))) t/s")
    }
}

#Preview {
    GenerationInfoView(tokensPerSecond: 58.5834)
}
