# Insta360StreamPlugin

Insta360 카메라의 라이브 스트림 기능을 이용해서 Unity에서 360 비디오를 실시간으로 보여주는 C++ 네이티브 플러그인입니다. 하드웨어 가속(CUDA, NPP)을 이용해 빠르고 안정적으로 동작합니다. (DX12 기반)

Insta360 X3로 테스트 되었습니다.

## 동작 원리

1.  **카메라 스레드 (네이티브)**: Insta360 카메라에서 H.264 스트림을 받아 GPU에서 디코딩하고, 색상을 변환합니다 (NV12 → BGR).
2.  **데이터 큐**: 처리된 영상 프레임을 잠시 저장합니다.
3.  **Unity 렌더 스레드**: 큐에서 프레임을 가져와 Unity 텍스처에 그려줍니다 (BGR → RGBA).
4.  **텍스처 업데이트 (C\#)**: `LoadRawTextureData`를 사용해 CPU에서 GPU 메모리로 이미지 데이터를 업로드하여 Unity 텍스처를 직접 업데이트합니다.

## 준비물

⚠️ **중요**: 아래 라이브러리들은 직접 다운로드해서 설치하고, Visual Studio 프로젝트에 경로를 설정해야 합니다.

  * Insta360 C++ Camera SDK
  * FFmpeg (CUDA, NPP 지원 빌드)
  * NVIDIA CUDA Toolkit 11.8

## 빌드하기

1.  이 저장소를 복제(clone)합니다.
2.  위에 명시된 준비물들을 모두 설치합니다.
3.  Visual Studio에서 `.sln` 파일을 열고, 프로젝트 속성에서 아래 경로들을 설정합니다.
      * **Include 경로**: `C/C++ -> 일반 -> 추가 포함 디렉터리`
      * **Library 경로**: `링커 -> 일반 -> 추가 라이브러리 디렉터리`
4.  프로젝트를 빌드하여 `RenderingPlugin.dll` 파일을 만듭니다.

## Unity에서 사용하기

1.  만들어진 `RenderingPlugin.dll` 파일을 Unity 프로젝트의 `Assets/Plugins/x86_64` 폴더에 넣습니다.
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
2.  씬(Scene)에 빈 게임 오브젝트를 하나 만들고, 제공된 `insta360.cs` 스크립트를 추가합니다.
3.  인스펙터(Inspector) 창에서 아래 항목들을 연결해 줍니다.
      * **Sphere**: 영상이 씌워질 구(Sphere) 모델 오브젝트
      * **Skybox Material**: 듀얼 피시아이 영상을 적용할 스카이박스 머티리얼

## 향후 과제

현재 처리 과정은 디코딩된 프레임 데이터를 GPU 메모리에서 CPU로 복사한 뒤, 텍스처 업데이트를 위해 다시 GPU로 업로드하는 방식을 사용합니다. 이러한 GPU → CPU → GPU 왕복 과정은 비효율적입니다.

향후 개발의 주요 목표는 cuda-d3d12 interoperability 를 구현하는 것입니다. 이를 통해 CUDA에서 디코딩된 GPU 리소스를 Direct3D 12 렌더링 컨텍스트와 직접 공유할 수 있게 됩니다. 불필요한 메모리 복사 과정을 생략하여 성능을 크게 향상시키고 지연 시간(latency)을 줄일 수 있습니다.

## 라이선스

  * 이 프로젝트의 기본 라이선스는 **MIT**입니다.
  * 단, Unity 공식 예제를 기반으로 했으므로 **Unity에서만 사용**할 수 있습니다 (Unity Companion License).
  * 또한, **Insta360 SDK**와 **FFmpeg**의 라이선스도 준수해야 합니다.