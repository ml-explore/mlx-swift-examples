// Copyright © 2026 Apple Inc.

import Foundation
import QuartzCore

enum RendererKind: String, CaseIterable, Identifiable {
    case cpu = "CPU"
    case mlx = "MLX"
    case mlxCompiled = "MLX Compiled"
    case mlxMetal = "MLX Metal"

    var id: String { rawValue }
}

@MainActor @Observable
class Renderer {

    public var image: IOSurface?
    public var configuration = Configuration()
    public var renderTime: TimeInterval?
    public var kind: RendererKind = .mlxCompiled {
        didSet {
            if kind != oldValue {
                recentTimes.removeAll(keepingCapacity: true)
                averageFrameTime = nil
                lastDisplayUpdate = 0
            }
        }
    }
    public private(set) var isAnimating = false

    private var renderTask: Task<Void, Never>?

    @ObservationIgnored
    private var recentTimes: [TimeInterval] = []
    private let recentTimesCapacity = 30

    @ObservationIgnored
    private var lastDisplayUpdate: CFTimeInterval = 0
    private let displayUpdateInterval: CFTimeInterval = 0.5

    public private(set) var averageFrameTime: TimeInterval?

    public var averageFPS: Double? {
        guard let t = averageFrameTime, t > 0 else { return nil }
        return 1.0 / t
    }

    // a little tour.  note that the final zoom is near the
    // limit of what can be computed using float32
    let steps: [(Float, Configuration)] = [
        (
            5,
            .init()
        ),
        (
            5,
            .init(
                centerX: -0.5,
                centerY: -0.75,
                zoom: 5,
            )
        ),
        (
            5,
            .init(
                centerX: -0.12,
                centerY: -1.05,
                zoom: 10,
            )
        ),
        (
            5,
            .init(
                centerX: -0.155,
                centerY: -1.03,
                zoom: 40,
            )
        ),
        (
            5,
            .init(
                centerX: -0.167,
                centerY: -1.041,
                zoom: 400,
            )
        ),
        (
            5,
            .init(
                centerX: -0.16707328273332678,
                centerY: -1.0409595252713189,
                zoom: 4000,
            )
        ),
    ]

    private var step = 0
    private var animationStart: CFTimeInterval = 0
    private var animationTimer: Timer?

    public func render() {
        renderTask = Task.detached { [self, configuration, kind] in
            let start = Date.timeIntervalSinceReferenceDate
            let result: IOSurface
            switch kind {
            case .cpu:
                result = renderMandelbrotCPU(configuration: configuration)
            case .mlx:
                result = renderMandelbrotMLX(
                    configuration: configuration, compute: computeMandelbrotMLX)
            case .mlxCompiled:
                result = renderMandelbrotMLX(
                    configuration: configuration, compute: computeMandelbrotMLXCompiled)
            case .mlxMetal:
                result = renderMandelbrotMLX(
                    configuration: configuration, compute: computeMandelbrotMetal)
            }
            let end = Date.timeIntervalSinceReferenceDate

            await MainActor.run {
                self.image = result
                self.renderTime = end - start
                self.recordFrameTime(end - start)
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

        step = 0
        animationStart = CACurrentMediaTime()
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
        guard renderTask == nil else { return }

        let current = steps[step]
        let duration = CFTimeInterval(current.0)
        let elapsed = CACurrentMediaTime() - animationStart

        if elapsed > duration {
            animationStart = CACurrentMediaTime()
            step += 1
            self.configuration = steps[step].1
            render()

            if step == steps.count - 1 {
                stopAnimation()
            }
            return
        }

        let t = Float(min(elapsed / duration, 1.0))

        let stepCount = duration * 60
        let substep = Int(t * Float(stepCount))

        let next = steps[step + 1]
        self.configuration = current.1.lerp(other: next.1, steps: Int(stepCount), step: substep)

        render()
    }
}
