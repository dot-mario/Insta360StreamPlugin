// Example low level rendering Unity plugin
#include "RenderingPluginInterface.h"

#include "PlatformBase.h"
#include "RenderAPI.h"

#include <assert.h>
#include <math.h>
#include <vector>


#include <queue>
#include <mutex>

#include <iostream>

static std::queue<BGRDataItem> g_BGRDataQueueFront;
static std::queue<BGRDataItem> g_BGRDataQueueBack;
static std::mutex g_BGRDataQueueMutexFront;
static std::mutex g_BGRDataQueueMutexBack;

void QueueBGRData(Npp8u* bgrData, int width, int height, int stride, int idx)
{
	BGRDataItem item = { bgrData, width, height, stride, idx};
	if (idx == 0)
	{
		std::lock_guard<std::mutex> lock(g_BGRDataQueueMutexFront);
		g_BGRDataQueueFront.push(item);
	}
	else if (idx == 1)
	{
		std::lock_guard<std::mutex> lock(g_BGRDataQueueMutexBack);
		g_BGRDataQueueBack.push(item);
	}
	else {
		std::cout << "[insta360][error] wrong idx: " << idx << "\n";
	}
}

// --------------------------------------------------------------------------
// SetTimeFromUnity, an example function we export which is called by one of the scripts.

static float g_Time;

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTimeFromUnity (float t) { g_Time = t; }



// --------------------------------------------------------------------------
// SetTextureFromUnity, an example function we export which is called by one of the scripts.

static void* g_TextureHandleFront = NULL;
static int   g_TextureWidth  = 0;
static int   g_TextureHeight = 0;
// added by CHS
static void* g_TextureHandleBack = NULL;
static int   g_TextureWidth2 = 0;
static int   g_TextureHeight2 = 0;

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(void* textureHandle, int w, int h)
{
	// A script calls this at initialization time; just remember the texture pointer here.
	// Will update texture pixels each frame from the plugin rendering event (texture update
	// needs to happen on the rendering thread).
	g_TextureHandleFront = textureHandle;
	g_TextureWidth = w;
	g_TextureHeight = h;
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTexture2FromUnity(void* textureHandle, int w, int h)
{
	// A script calls this at initialization time; just remember the texture pointer here.
	// Will update texture pixels each frame from the plugin rendering event (texture update
	// needs to happen on the rendering thread).
	g_TextureHandleBack = textureHandle;
	g_TextureWidth2 = w;
	g_TextureHeight2 = h;
}


// --------------------------------------------------------------------------
// SetMeshBuffersFromUnity, an example function we export which is called by one of the scripts.

static void* g_VertexBufferHandle = NULL;
static int g_VertexBufferVertexCount;

struct MeshVertex
{
	float pos[3];
	float normal[3];
	float color[4];
	float uv[2];
};
static std::vector<MeshVertex> g_VertexSource;


extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetMeshBuffersFromUnity(void* vertexBufferHandle, int vertexCount, float* sourceVertices, float* sourceNormals, float* sourceUV)
{
	// A script calls this at initialization time; just remember the pointer here.
	// Will update buffer data each frame from the plugin rendering event (buffer update
	// needs to happen on the rendering thread).
	g_VertexBufferHandle = vertexBufferHandle;
	g_VertexBufferVertexCount = vertexCount;

	// The script also passes original source mesh data. The reason is that the vertex buffer we'll be modifying
	// will be marked as "dynamic", and on many platforms this means we can only write into it, but not read its previous
	// contents. In this example we're not creating meshes from scratch, but are just altering original mesh data --
	// so remember it. The script just passes pointers to regular C# array contents.
	g_VertexSource.resize(vertexCount);
	for (int i = 0; i < vertexCount; ++i)
	{
		MeshVertex& v = g_VertexSource[i];
		v.pos[0] = sourceVertices[0];
		v.pos[1] = sourceVertices[1];
		v.pos[2] = sourceVertices[2];
		v.normal[0] = sourceNormals[0];
		v.normal[1] = sourceNormals[1];
		v.normal[2] = sourceNormals[2];
		v.uv[0] = sourceUV[0];
		v.uv[1] = sourceUV[1];
		sourceVertices += 3;
		sourceNormals += 3;
		sourceUV += 2;
	}
}


// --------------------------------------------------------------------------
// UnitySetInterfaces

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);

static IUnityInterfaces* s_UnityInterfaces = NULL;
static IUnityGraphics* s_Graphics = NULL;

//static IUnityInterfaces* s_UnityInterfaces2 = NULL;
//static IUnityGraphics* s_Graphics2 = NULL;

extern "C" void	UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces)
{
	s_UnityInterfaces = unityInterfaces;
	s_Graphics = s_UnityInterfaces->Get<IUnityGraphics>();
	s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

#if SUPPORT_VULKAN
	if (s_Graphics->GetRenderer() == kUnityGfxRendererNull)
	{
		extern void RenderAPI_Vulkan_OnPluginLoad(IUnityInterfaces*);
		RenderAPI_Vulkan_OnPluginLoad(unityInterfaces);
	}
#endif // SUPPORT_VULKAN

	// Run OnGraphicsDeviceEvent(initialize) manually on plugin load
	OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
{
	s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
}

#if UNITY_WEBGL
typedef void	(UNITY_INTERFACE_API * PluginLoadFunc)(IUnityInterfaces* unityInterfaces);
typedef void	(UNITY_INTERFACE_API * PluginUnloadFunc)();

extern "C" void	UnityRegisterRenderingPlugin(PluginLoadFunc loadPlugin, PluginUnloadFunc unloadPlugin);

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API RegisterPlugin()
{
	UnityRegisterRenderingPlugin(UnityPluginLoad, UnityPluginUnload);
}
#endif

// --------------------------------------------------------------------------
// GraphicsDeviceEvent


static RenderAPI* s_CurrentAPI = NULL;
static UnityGfxRenderer s_DeviceType = kUnityGfxRendererNull;

//static RenderAPI* s_CurrentAPI2 = NULL; // To prevent from bottle neck when Updating Textures only with one RenderAPIz
//static UnityGfxRenderer s_DeviceType2 = kUnityGfxRendererNull;


static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
	// Create graphics API implementation upon initialization
	if (eventType == kUnityGfxDeviceEventInitialize)
	{
		assert(s_CurrentAPI == NULL);
		s_DeviceType = s_Graphics->GetRenderer();
		s_CurrentAPI = CreateRenderAPI(s_DeviceType);
	}

	// Let the implementation process the device related events
	if (s_CurrentAPI)
	{
		s_CurrentAPI->ProcessDeviceEvent(eventType, s_UnityInterfaces);
	}

	// Cleanup graphics API implementation upon shutdown
	if (eventType == kUnityGfxDeviceEventShutdown)
	{
		delete s_CurrentAPI;
		s_CurrentAPI = NULL;
		s_DeviceType = kUnityGfxRendererNull;
	}
}

void UpdateFrontTextureFromBGRDataCPU(Npp8u* bgrData, int width, int height, int stride)
{
	//std::cout << "[insta360][debug] UpdateFrontTextureFromBGRDataCPU 호출됨" << std::endl;
	//	<< width << ", height: " << height << ", stride: " << stride << "\n";

	size_t dataSize = width * height * 3;  // 3채널 BGR 데이터 크기
	//std::cout << "[insta360][debug] 계산된 데이터 사이즈: " << dataSize << "\n";

	unsigned char* hostBGR = (unsigned char*)malloc(dataSize);
	if (!hostBGR)
	{
		std::cout << "[insta360][error] 호스트 버퍼 할당 실패" << "\n";
		return;
	}

	void* textureHandle = g_TextureHandleFront;
	//std::cout << "[insta360][debug] g_TextureHandleFront = " << textureHandle << "\n";
	if (!textureHandle || !s_CurrentAPI)
	{
		std::cout << "[insta360][error] Texture handle 또는 RenderAPI가 유효하지 않음" << "\n";
		free(hostBGR);
		return;
	}

	int textureRowPitch = 0;
	//std::cout << "[insta360][debug] BeginModifyTexture 호출 전" << "\n";
	void* textureDataPtr = s_CurrentAPI->BeginModifyTexture(textureHandle, width, height, &textureRowPitch);
	if (!textureDataPtr)
	{
		std::cout << "[insta360][error] BeginModifyTexture 실패" << "\n";
		free(hostBGR);
		return;
	}
	//std::cout << "[insta360][debug] BeginModifyTexture 성공 - Row Pitch: " << textureRowPitch << "\n";

	cudaError_t cudaStatus = cudaMemcpy(hostBGR, bgrData, dataSize, cudaMemcpyDeviceToHost);;
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "[insta360][error] cudaMemcpyDeviceToHost 실패, error code: "
			<< cudaStatus << "\n";
		free(hostBGR);
		return;
	}

	memset(textureDataPtr, 0, height * textureRowPitch);
	//std::cout << "[insta360][debug] Texture 메모리 클리어 완료" << "\n";

	// BGR -> RGBA 변환
	unsigned char* dstPtr = (unsigned char*)textureDataPtr;
	for (int y = 0; y < height; ++y)
	{
		unsigned char* dstRow = dstPtr + y * width * 4;
		unsigned char* srcRow = hostBGR + (height - 1 - y) * width * 3;
		for (int x = 0; x < width; ++x)
		{
			dstRow[x * 4 + 0] = srcRow[x * 3 + 0]; // Red
			dstRow[x * 4 + 1] = srcRow[x * 3 + 1]; // Green
			dstRow[x * 4 + 2] = srcRow[x * 3 + 2]; // Blue
			dstRow[x * 4 + 3] = 255;               // Alpha
		}
	}

	s_CurrentAPI->EndModifyTexture(textureHandle, width, height, textureRowPitch, textureDataPtr);
	//std::cout << "[insta360][debug] EndModifyTexture 호출 완료" << "\n";

	free(hostBGR);
}

void UpdateBackTextureFromBGRDataCPU(Npp8u* bgrData, int width, int height, int stride)
{
	//std::cout << "[insta360][debug] UpdateBackTextureFromBGRDataCPU 호출됨" << std::endl;
	//	<< width << ", height: " << height << ", stride: " << stride << "\n";

	size_t dataSize = width * height * 3;  // 3채널 BGR 데이터 크기
	//std::cout << "[insta360][debug] 계산된 데이터 사이즈: " << dataSize << "\n";

	unsigned char* hostBGR = (unsigned char*)malloc(dataSize);
	if (!hostBGR)
	{
		std::cout << "[insta360][error] 호스트 버퍼 할당 실패" << "\n";
		return;
	}

	void* textureHandle = g_TextureHandleBack;
	//std::cout << "[insta360][debug] g_TextureHandleFront = " << textureHandle << "\n";
	if (!textureHandle || !s_CurrentAPI)
	{
		std::cout << "[insta360][error] Texture handle 또는 RenderAPI가 유효하지 않음" << "\n";
		free(hostBGR);
		return;
	}

	int textureRowPitch = 0;
	//std::cout << "[insta360][debug] BeginModifyTexture2 호출 전" << "\n";
	void* textureDataPtr = s_CurrentAPI->BeginModifyTexture2(textureHandle, width, height, &textureRowPitch);
	if (!textureDataPtr)
	{
		std::cout << "[insta360][error] BeginModifyTexture 실패" << "\n";
		free(hostBGR);
		return;
	}
	//std::cout << "[insta360][debug] BeginModifyTexture2 성공 - Row Pitch: " << textureRowPitch << "\n";

	cudaError_t cudaStatus = cudaMemcpy(hostBGR, bgrData, dataSize, cudaMemcpyDeviceToHost);;
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "[insta360][error] cudaMemcpyDeviceToHost 실패, error code: "
			<< cudaStatus << "\n";
		free(hostBGR);
		return;
	}

	memset(textureDataPtr, 0, height * textureRowPitch);
	//std::cout << "[insta360][debug] Texture 메모리 클리어 완료" << "\n";

	// BGR -> RGBA 변환
	unsigned char* dstPtr = (unsigned char*)textureDataPtr;
	for (int y = 0; y < height; ++y)
	{
		unsigned char* dstRow = dstPtr + y * width * 4;
		unsigned char* srcRow = hostBGR + (height - 1 - y) * width * 3;
		for (int x = 0; x < width; ++x)
		{
			dstRow[x * 4 + 0] = srcRow[x * 3 + 0]; // Red
			dstRow[x * 4 + 1] = srcRow[x * 3 + 1]; // Green
			dstRow[x * 4 + 2] = srcRow[x * 3 + 2]; // Blue
			dstRow[x * 4 + 3] = 255;               // Alpha
		}
	}

	s_CurrentAPI->EndModifyTexture2(textureHandle, width, height, textureRowPitch, textureDataPtr);
	//std::cout << "[insta360][debug] EndModifyTexture2 호출 완료" << "\n";

	free(hostBGR);
}

static void drawToPluginTexture()
{
	s_CurrentAPI->drawToPluginTexture();
}

static void drawToRenderTexture()
{
	s_CurrentAPI->drawToRenderTexture();
}

static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
	// Unknown / unsupported graphics device type? Do nothing
	if (s_CurrentAPI == NULL)
		return;

	if (eventID == 1)
	{
        //drawToRenderTexture();
        //DrawColoredTriangle();
        //ModifyTexturePixels();
        //ModifyVertexBuffer();

		// front 스트림 처리
		BGRDataItem frontImage;
		bool frontAvailable = false;
		{
			std::lock_guard<std::mutex> lock(g_BGRDataQueueMutexFront);
			if (!g_BGRDataQueueFront.empty())
			{
				frontImage = g_BGRDataQueueFront.front();
				g_BGRDataQueueFront.pop();
				frontAvailable = true;
			}
		}
		// 락 해제 후, 오래 걸리는 업데이트 수행
		if (frontAvailable)
		{
			UpdateFrontTextureFromBGRDataCPU(frontImage.data, frontImage.width, frontImage.height, frontImage.stride);
			nppiFree(frontImage.data);
		}

		// back 스트림 처리
		BGRDataItem backImage;
		bool backAvailable = false;
		{
			std::lock_guard<std::mutex> lock(g_BGRDataQueueMutexBack);
			if (!g_BGRDataQueueBack.empty())
			{
				backImage = g_BGRDataQueueBack.front();
				g_BGRDataQueueBack.pop();
				backAvailable = true;
			}
		}
		if (backAvailable)
		{
			UpdateBackTextureFromBGRDataCPU(backImage.data, backImage.width, backImage.height, backImage.stride);
			nppiFree(backImage.data);
		}
	}

	if (eventID == 2)
	{
		// d3d12만을 위한 코드로, 플러그인이 관리, 출력하는 텍스처를 위한 함수이므로 현재 우리에겐 필요없음
		//drawToPluginTexture();
	}

	//if (eventID == 3)
	//{
	//	std::lock_guard<std::mutex> lock(g_BGRDataQueueMutexBack);
	//	if (!g_BGRDataQueueBack.empty())
	//	{
	//		BGRDataItem backImage = g_BGRDataQueueBack.front();
	//		g_BGRDataQueueBack.pop();
	//		/*std::cout << "[insta360][debug] Processing back image: "
	//			<< backImage.width << "x" << backImage.height
	//			<< ", stride: " << backImage.stride << "\n";*/
	//		UpdateBackTextureFromBGRDataCPU(backImage.data, backImage.width, backImage.height, backImage.stride);
	//		nppiFree(backImage.data);
	//	}
	//}

	//if (eventID == 4)
	//{
	//	drawToPluginTexture2();
	//}

}

// --------------------------------------------------------------------------
// `RenderEventFunc, an example function we export which is used to get a rendering event callback function.

extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc()
{
	return OnRenderEvent;
}

// --------------------------------------------------------------------------
// DX12 plugin specific
// --------------------------------------------------------------------------

extern "C" UNITY_INTERFACE_EXPORT void* UNITY_INTERFACE_API GetRenderTexture()
{
	return s_CurrentAPI->getRenderTexture();
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetRenderTexture(UnityRenderBuffer rb)
{
	s_CurrentAPI->setRenderTextureResource(rb);
}

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsSwapChainAvailable()
{
	return s_CurrentAPI->isSwapChainAvailable();
}

extern "C" UNITY_INTERFACE_EXPORT unsigned int UNITY_INTERFACE_API GetPresentFlags()
{
	return s_CurrentAPI->getPresentFlags();
}

extern "C" UNITY_INTERFACE_EXPORT unsigned int UNITY_INTERFACE_API GetSyncInterval()
{
	return s_CurrentAPI->getSyncInterval();
}

extern "C" UNITY_INTERFACE_EXPORT unsigned int UNITY_INTERFACE_API GetBackBufferWidth()
{
	return s_CurrentAPI->getBackbufferHeight();
}

extern "C" UNITY_INTERFACE_EXPORT unsigned int UNITY_INTERFACE_API GetBackBufferHeight()
{
	return s_CurrentAPI->getBackbufferWidth();
}
