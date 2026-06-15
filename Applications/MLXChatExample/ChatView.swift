//
//  ChatView.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import AVFoundation
import AVKit
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

#if canImport(UIKit)
    import UIKit

    /// Transferable wrapper that explicitly requests image content type from PhotosPicker.
    private struct PickedImage: Transferable {
        let data: Data

        static var transferRepresentation: some TransferRepresentation {
            DataRepresentation(importedContentType: .image) { data in
                PickedImage(data: data)
            }
        }
    }

    /// Transferable wrapper for video content from PhotosPicker.
    private struct PickedVideo: Transferable {
        let url: URL

        static var transferRepresentation: some TransferRepresentation {
            FileRepresentation(importedContentType: .movie) { receivedFile in
                let dest = FileManager.default.temporaryDirectory
                    .appendingPathComponent(
                        "\(UUID().uuidString).\(receivedFile.file.pathExtension)")
                try FileManager.default.copyItem(at: receivedFile.file, to: dest)
                return PickedVideo(url: dest)
            }
        }
    }
#endif

/// Main chat interface view that manages the conversation UI and user interactions.
/// Displays messages, handles media attachments, and provides input controls.
struct ChatView: View {
    /// View model that manages the chat state and business logic
    @Bindable private var vm: ChatViewModel

    #if os(iOS)
        /// Selected items from PhotosPicker
        @State private var photosPickerItems: [PhotosPickerItem] = []
    #endif

    /// Initializes the chat view with a view model
    /// - Parameter viewModel: The view model to manage chat state
    init(viewModel: ChatViewModel) {
        self.vm = viewModel
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Display conversation history
                ConversationView(messages: vm.messages)

                Divider()

                // Show media previews if attachments are present
                if !vm.mediaSelection.isEmpty {
                    MediaPreviewsView(mediaSelection: vm.mediaSelection)
                }

                // Input field with send and media attachment buttons
                PromptField(
                    prompt: $vm.prompt,
                    sendButtonAction: vm.generate,
                    // Only show media button for vision-capable models
                    mediaButtonAction: vm.selectedModel.isVisionModel
                        ? {
                            vm.mediaSelection.isShowing = true
                        } : nil
                )
                .padding()
            }
            .navigationTitle("MLX Chat Example")
            .toolbar {
                ChatToolbarView(vm: vm)
            }
            // Handle media file selection
            #if os(iOS)
                .photosPicker(
                    isPresented: $vm.mediaSelection.isShowing,
                    selection: $photosPickerItems,
                    maxSelectionCount: 1,
                    matching: .any(of: [.images, .videos])
                )
                .onChange(of: photosPickerItems) {
                    Task {
                        for item in photosPickerItems {
                            if item.supportedContentTypes.contains(where: {
                                $0.conforms(to: .image)
                            }) {
                                // Load image with explicit .image content type
                                if let picked = try? await item.loadTransferable(
                                    type: PickedImage.self),
                                    let uiImage = UIImage(data: picked.data)
                                {
                                    // Normalize orientation so pixels match the display orientation.
                                    // UIImage.jpegData() only writes an EXIF tag but CIImage(contentsOf:)
                                    // does not apply it, so the VLM would receive a rotated image.
                                    let renderer = UIGraphicsImageRenderer(size: uiImage.size)
                                    let oriented = renderer.image { _ in
                                        uiImage.draw(in: CGRect(origin: .zero, size: uiImage.size))
                                    }
                                    if let jpegData = oriented.jpegData(compressionQuality: 0.9) {
                                        let url =
                                            FileManager.default.temporaryDirectory
                                        .appendingPathComponent("\(UUID().uuidString).jpg")
                                        try? jpegData.write(to: url)
                                        vm.addMedia(.success(url))
                                    }
                                }
                            } else if item.supportedContentTypes.contains(where: {
                                $0.conforms(to: .movie)
                            }) {
                                // Load video with explicit .movie content type
                                if let picked = try? await item.loadTransferable(
                                    type: PickedVideo.self)
                                {
                                    vm.addMedia(.success(picked.url))
                                }
                            }
                        }
                        photosPickerItems = []
                    }
                }
            #else
                .fileImporter(
                    isPresented: $vm.mediaSelection.isShowing,
                    allowedContentTypes: [.image, .movie],
                    onCompletion: vm.addMedia
                )
            #endif
        }
    }
}

#Preview {
    ChatView(viewModel: ChatViewModel(mlxService: MLXService()))
}
