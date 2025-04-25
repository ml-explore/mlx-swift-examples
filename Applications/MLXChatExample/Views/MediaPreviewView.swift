//
//  MediaPreviewView.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 21.04.2025.
//

import AVFoundation
import AVKit
import SwiftUI

/// A view that displays a horizontal scrollable list of media previews (images and videos).
struct MediaPreviewsView: View {
    /// The media selection containing arrays of image and video URLs
    let mediaSelection: MediaSelection

    var body: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 8) {
                // Display image previews
                ForEach(mediaSelection.images, id: \.self) { imageURL in
                    MediaPreviewView(
                        mediaURL: imageURL,
                        type: .image,
                        onRemove: {
                            mediaSelection.images.removeAll(where: { $0 == imageURL })
                        }
                    )
                }

                // Display video previews
                ForEach(mediaSelection.videos, id: \.self) { videoURL in
                    MediaPreviewView(
                        mediaURL: videoURL,
                        type: .video,
                        onRemove: {
                            mediaSelection.videos.removeAll(where: { $0 == videoURL })
                        }
                    )
                }
            }
            .padding(.horizontal)
        }
        .padding(.top)
    }
}

/// A view that displays a single media item (image or video) with a remove button.
struct MediaPreviewView: View {
    /// URL of the media file to display
    let mediaURL: URL
    /// Type of media (image or video)
    let type: MediaPreviewType
    /// Callback to handle removal of the media item
    let onRemove: () -> Void

    var body: some View {
        ZStack(alignment: .topTrailing) {
            switch type {
            case .image:
                // Display image with loading placeholder
                AsyncImage(url: mediaURL) { image in
                    image
                        .resizable()
                        .scaledToFit()
                        .frame(height: 100)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                } placeholder: {
                    ProgressView()
                        .frame(width: 150, height: 100)
                }
            case .video:
                // Display video player
                VideoPlayer(player: AVPlayer(url: mediaURL))
                    .frame(width: 150, height: 100)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            RemoveButton(action: onRemove)
        }
    }
}

/// A button for removing media items from the preview.
struct RemoveButton: View {
    /// Action to perform when the remove button is tapped
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: "xmark.circle.fill")
                .foregroundStyle(.secondary)
                .imageScale(.large)
        }
        .buttonStyle(.plain)
        .padding(4)
    }
}

extension MediaPreviewView {
    /// Defines the type of media that can be displayed in the preview
    enum MediaPreviewType {
        /// An image file
        case image
        /// A video file
        case video
    }
}

#Preview("Remove Button") {
    RemoveButton {}
}
