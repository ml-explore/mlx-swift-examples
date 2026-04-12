// Copyright © 2024 Apple Inc.

import Foundation
import XCTest

@testable import MLX
@testable import MLXNN

@testable import Gemma4Audio

class Gemma4AudioTests: XCTestCase {

    override class func setUp() {
        MLX.Device.setDefault(device: .gpu)
    }

    func testAudioConfig() {
        let config = AudioConfig()
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.numAttentionHeads, 8)
    }

    func testAudioRMSNorm() {
        let norm = AudioRMSNorm(dim: 1024)
        let x = MLXArray.ones([2, 100, 1024])
        let out = norm(x)
        XCTAssertEqual(out.shape, [2, 100, 1024])
    }

    func testSSCPConvBlock() {
        let block = SSCPConvBlock(inChannels: 1, outChannels: 256)
        let x = MLXArray.ones([2, 100, 128, 1])
        let mask = MLXArray.zeros([2, 100]).asType(.bool)
        let (out, outMask) = block(x, mask: mask)
        // With stride 2, the time and freq dimensions are approximately halved.
        // T_out = (T + 2*pad - kernel)/stride + 1 = (100 + 2 - 3)/2 + 1 = 50
        // F_out = (128 + 2 - 3)/2 + 1 = 64
        XCTAssertEqual(out.dim(0), 2)
        XCTAssertEqual(out.dim(1), 50)
        XCTAssertEqual(out.dim(2), 64)
        XCTAssertEqual(out.dim(3), 256)
        XCTAssertEqual(outMask.shape, [2, 50])
    }

    func testSubSampleConvProjection() {
        let config = AudioConfig()
        let sscp = SubSampleConvProjection(
            hiddenSize: config.hiddenSize, subsamplingConvChannels: config.subsamplingConvChannels,
            rmsNormEps: config.rmsNormEps)
        let x = MLXArray.ones([2, 100, 128])
        let mask = MLXArray.zeros([2, 100]).asType(.bool)
        let (out, outMask) = sscp(x, mask: mask)
        XCTAssertEqual(out.dim(0), 2)
        XCTAssertEqual(out.dim(1), 25)
        XCTAssertEqual(out.dim(2), 1024)  // Projected to hiddenSize
        XCTAssertEqual(outMask.shape, [2, 25])
    }

    func testAudioEncoderShape() {
        // Use a smaller config to speed up the test
        let config = AudioConfig(
            hiddenSize: 128,
            numAttentionHeads: 4,
            convKernelSize: 31,
            subsamplingConvChannels: [64, 64],
            attentionContextLeft: 32,
            attentionContextRight: 0,
            attentionChunkSize: 32
        )
        let encoder = AudioEncoder(config: config, numHiddenLayers: 2)
        let x = MLXArray.ones([2, 100, 128])
        let mask = MLXArray.zeros([2, 100]).asType(.bool)

        let (out, outMask) = encoder(x, mask: mask)
        XCTAssertEqual(out.dim(0), 2)
        XCTAssertEqual(out.dim(1), 25)
        XCTAssertEqual(out.dim(2), 128)
    }

    func testAudioFeatureExtractor() {
        let extractor = Gemma4AudioFeatureExtractor()
        // 1 second of audio at 16kHz
        let rawSpeech = MLXArray.ones([16000])

        let (features, mask) = extractor(rawSpeech: [rawSpeech])

        XCTAssertEqual(features.dim(0), 1)
        XCTAssertEqual(features.dim(2), 128)  // feature size
        XCTAssertEqual(mask.dim(0), 1)
        // 16000 samples / 160 samples per hop (10ms) = 100 frames
        XCTAssertEqual(features.dim(1), mask.dim(1))
    }
}
