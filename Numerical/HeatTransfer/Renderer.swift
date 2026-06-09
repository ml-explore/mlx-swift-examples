// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import QuartzCore

private let iterationsPerTick = 200

enum RendererKind: String, CaseIterable, Identifiable {
    case conv2d = "Conv2D"
    case roll = "Roll"
    case sor = "SOR"
    case sorFullSpeed = "SOR Full Speed"

    var id: String { rawValue }

    var iterations: Int {
        switch self {
        case .conv2d, .roll: 200
        case .sor: 1
        case .sorFullSpeed: 200
        }
    }

    func apply(room: inout Room) {
        switch self {
        case .conv2d:
            computeJacobiConv2d(state: &room, count: self.iterations)
        case .roll:
            computeJacobiStencil(state: &room, count: self.iterations)
        case .sor, .sorFullSpeed:
            computeSOR(state: &room, count: self.iterations)
        }
    }
}

@MainActor @Observable
class Renderer {

    public var configuration: Configuration
    private var room: Room?
    public var kind: RendererKind = .conv2d {
        didSet {
            if kind != oldValue {
                recentTimes.removeAll(keepingCapacity: true)
                averageFrameTime = nil
                lastDisplayUpdate = 0
            }
        }
    }

    public var image: IOSurface?

    private var animationTimer: Timer?
    public private(set) var isAnimating = false
    private var renderTask: Task<Void, Never>?

    /// smoothed average render time per iteration.  Note: each visible frame may be
    /// several iterations.  See ``RendererKind/iterations``.
    public private(set) var averageFrameTime: TimeInterval?

    /// FPS as reciprocal of the frame time -- this is the maximum rate it could run.
    public var averageFPS: Double? {
        guard let t = averageFrameTime, t > 0 else { return nil }
        return 1.0 / t
    }

    @ObservationIgnored
    private var recentTimes: [TimeInterval] = []
    private let recentTimesCapacity = 30

    @ObservationIgnored
    private var lastDisplayUpdate: CFTimeInterval = 0
    private let displayUpdateInterval: CFTimeInterval = 0.5

    init() {
        var c = Configuration()
        for _ in 0 ..< Int.random(in: 6 ... 12) {
            c.addRandomWall()
        }
        for _ in 0 ..< Int.random(in: 8 ... 12) {
            c.addRandomHeatSource()
        }
        self.configuration = c
    }

    public func start() {
        let room = self.room ?? configuration.asRoom()

        renderTask = Task.detached { [self] in
            let wallColor = full(room.temperature.shape, values: UInt32(0xff80_8080))
                .view(dtype: .uint8)
                .reshaped(room.temperature.shape + [-1])
            var raster = applyLUT(
                room.temperature, lut: MLXArray(lut), max: 1.0, maxValue: 0xffff_ffff)
            raster = which(room.wallMask.expandedDimensions(axis: -1), wallColor, raster)
            let image = createIOSurface(bgra: raster)

            await MainActor.run {
                self.image = image
                self.room = room
                self.renderTask = nil
            }
        }
    }

    public func rebuildRoom() {
        var c = Configuration()
        for _ in 0 ..< Int.random(in: 6 ... 12) {
            c.addRandomWall()
        }
        for _ in 0 ..< Int.random(in: 8 ... 12) {
            c.addRandomHeatSource()
        }
        self.configuration = c
        self.room = nil
        self.recentTimes.removeAll(keepingCapacity: true)
        self.averageFrameTime = nil
        self.lastDisplayUpdate = 0
        start()
    }

    public func render() {
        var room = self.room ?? configuration.asRoom()

        renderTask = Task.detached { [self, kind] in
            // compute
            let start = Date.timeIntervalSinceReferenceDate

            // important: evaluate the result, otherwise
            // the time for the computation lands below when
            // we produce an image from it.
            kind.apply(room: &room)
            eval(room.temperature)

            let end = Date.timeIntervalSinceReferenceDate
            let timePerIteration = (end - start) / Double(kind.iterations)

            // render -- mix the room temperature with the walls
            let wallColor = full(room.temperature.shape, values: UInt32(0xff80_8080))
                .view(dtype: .uint8)
                .reshaped(room.temperature.shape + [-1])
            var raster = applyLUT(
                room.temperature, lut: MLXArray(lut), max: 1.0, maxValue: 0xffff_ffff)
            raster = which(room.wallMask.expandedDimensions(axis: -1), wallColor, raster)
            let image = createIOSurface(bgra: raster)

            await MainActor.run {
                self.image = image
                self.room = room
                self.recordFrameTime(timePerIteration)
                self.renderTask = nil
            }
        }
    }

    private func recordFrameTime(_ time: TimeInterval) {
        recentTimes.append(time)
        if recentTimes.count > recentTimesCapacity {
            recentTimes.removeFirst(recentTimes.count - recentTimesCapacity)
        }

        let now = CACurrentMediaTime()
        if now - lastDisplayUpdate >= displayUpdateInterval {
            lastDisplayUpdate = now
            averageFrameTime = recentTimes.reduce(0, +) / TimeInterval(recentTimes.count)
        }
    }

    public func startAnimation() {
        stopAnimation()
        isAnimating = true

        animationTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) {
            [weak self] _ in
            Task { @MainActor in
                self?.tick()
            }
        }
    }

    public func stopAnimation() {
        animationTimer?.invalidate()
        animationTimer = nil
        isAnimating = false
    }

    public func tick() {
        if renderTask == nil {
            render()
        }
    }
}
