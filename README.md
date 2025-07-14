# Insta360StreamPlugin

This is a C++ native plugin that uses the live stream feature of Insta360 cameras to display 360 video in Unity in real-time. It's fast and stable, using hardware acceleration (CUDA, NPP). (Based on DX12)

Tested with Insta360 X3.

## How it Works

1.  **Camera Thread (Native)**: Receives the H.264 stream from the Insta360 camera, decodes it on the GPU, and converts the color space (NV12 → BGR).
2.  **Data Queue**: Temporarily stores the processed video frames.
3.  **Unity Render Thread**: Pulls frames from the queue and prepares them for the Unity texture (BGR → RGBA).
4.  **Texture Update (C#)**: Uploads the image data from CPU to GPU memory using `LoadRawTextureData` to directly update the Unity texture.

## Prerequisites

⚠️ **Important**: You must download and install the following libraries yourself, then configure the paths in the Visual Studio project.

* Insta360 C++ Camera SDK
* FFmpeg (built with CUDA and NPP support)
* NVIDIA CUDA Toolkit

## Build

1.  Clone this repository.
2.  Install all the prerequisites listed above.
3.  Open the `.sln` file in Visual Studio and configure the following paths in the project properties:
    * **Include Directories**: `C/C++ -> General -> Additional Include Directories`
    * **Library Directories**: `Linker -> General -> Additional Library Directories`
4.  Build the project to create the `RenderingPlugin.dll` file.

## Usage in Unity

1.  Copy the built `RenderingPlugin.dll` file into your Unity project's `Assets/Plugins/x86_64` folder.
```markdown
Plugins/
├── RenderingPlugin.dll
├── CameraSDK.dll
├── avcodec-61.dll
├── avdevice-61.dll
├── avfilter-10.dll
├── avformat-61.dll
├── avutil-59.dll
├── postproc-58.dll
├── swresample-5.dll
└── swscale-8.dll
```
2.  Create an empty GameObject in the Scene and add the provided `insta360.cs` script to it.
3.  In the Inspector window, link the following components:
    * **Sphere**: The Sphere model object the video will be mapped onto.
    * **Skybox Material**: The Skybox material that will use the dual fisheye video textures.

## Future Work

The current process copies the decoded frame data from GPU memory to the CPU, only to upload it back to the GPU for the texture update. This GPU → CPU → GPU roundtrip is inefficient.

The primary goal for future development is to implement cuda-d3d12 interoperability. This will allow the decoded GPU resource from CUDA to be shared directly with the Direct3D 12 rendering context. This will eliminate the unnecessary memory copy, significantly improving performance and reducing latency.

## License

* The primary license for this project is **MIT**.
* However, since it's based on an official Unity example, it can **only be used with Unity** (per the Unity Companion License).
* You must also comply with the licenses for the **Insta360 SDK** and **FFmpeg**.