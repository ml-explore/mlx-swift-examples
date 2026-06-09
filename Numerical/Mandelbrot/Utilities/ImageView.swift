// Copyright © 2026 Apple Inc.

import SwiftUI

/// Internal view to display a CVImageBuffer
#if os(iOS)
    public struct ImageView: UIViewRepresentable {

        public let image: Any
        public var gravity = CALayerContentsGravity.resizeAspect

        public func makeUIView(context: Context) -> UIView {
            let view = UIView()
            view.layer.contentsGravity = gravity
            view.autoresizingMask = [.width, .height]
            return view
        }

        public func updateUIView(_ uiView: UIView, context: Context) {
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            uiView.layer.contents = image
            CATransaction.commit()
        }

    }
#else
    public struct ImageView: NSViewRepresentable {

        public let image: Any
        public var gravity = CALayerContentsGravity.resizeAspect

        public func makeNSView(context: Context) -> some NSView {
            let view = NSView()
            view.wantsLayer = true
            view.layer?.contentsGravity = gravity
            view.autoresizingMask = [.width, .height]
            return view
        }

        public func updateNSView(_ nsView: NSViewType, context: Context) {
            if let layer = nsView.layer {
                CATransaction.begin()
                CATransaction.setDisableActions(true)
                layer.contents = image
                CATransaction.commit()
            }
        }
    }
#endif
