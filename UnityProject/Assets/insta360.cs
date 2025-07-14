using UnityEngine;
using System;
using System.Collections;
using System.Runtime.InteropServices;
using UnityEngine.Rendering;
using System.Diagnostics.CodeAnalysis;

public class insta360 : MonoBehaviour
{
    // 플러그인 함수 선언
#if (PLATFORM_IOS || PLATFORM_TVOS || PLATFORM_BRATWURST || PLATFORM_SWITCH) && !UNITY_EDITOR
    [DllImport("__Internal")]
#else
    [DllImport("RenderingPlugin")]
#endif
    private static extern void SetTimeFromUnity(float t);

#if (PLATFORM_IOS || PLATFORM_TVOS || PLATFORM_BRATWURST || PLATFORM_SWITCH) && !UNITY_EDITOR
    [DllImport("__Internal")]
#else
    [DllImport("RenderingPlugin")]
#endif
    private static extern IntPtr GetRenderEventFunc();

#if PLATFORM_SWITCH && !UNITY_EDITOR
    [DllImport("__Internal")]
    private static extern void RegisterPlugin();
#endif

#if (UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || UNITY_WSA || UNITY_WSA_10_0)
    [DllImport("RenderingPlugin")]
    private static extern void StartStream();

    [DllImport("RenderingPlugin")]
    private static extern void StopStream();

    [DllImport("RenderingPlugin")]
    private static extern void SetVideoFilePath(string path);

    [DllImport("RenderingPlugin")]
    private static extern void StopSaveVideo();

    [DllImport("RenderingPlugin")]
    private static extern bool GetNextRawVideoPacket(int streamIndex, out IntPtr dataPtr, out int size);

    [DllImport("RenderingPlugin")]
    private static extern void FreeRawVideoPacket(IntPtr dataPtr);

    [DllImport("RenderingPlugin")]
    private static extern int GetDecodedPacket(byte[] inData, int inSize, int streamIndex, out IntPtr outData,
        out int outSize);

    [DllImport("RenderingPlugin")]
    private static extern int InitFfmpegDecoder();
    
    [SerializeField] private int width = 1280;
    [SerializeField] private int height = 1280;
    [SerializeField] private string videoFilePath;
    
    // 성능 최적화를 위한 설정
    [SerializeField] private int textureUpdateInterval = 1;
    private int frameCounter = 0;
    
    // 메모리 모니터링 설정
    [SerializeField] private float maxRuntimeMinutes = 60f;
    private const float MEMORY_CHECK_INTERVAL = 10f;
    private const float MAX_MEMORY_THRESHOLD = 1024f; // MB 단위

    public GameObject sphere;
    public Material skyboxMaterial;
    
    private Texture2D frontTexture;
    private Texture2D backTexture;
    
    // 코루틴 참조
    private Coroutine pluginUpdateCoroutine;
#else
    private void SetRenderTexture(IntPtr rb) { }
#endif

    private void Awake()
    {
        SetStreamTexture();
    }

    IEnumerator Start()
    {
#if PLATFORM_SWITCH && !UNITY_EDITOR
        RegisterPlugin();
#endif
        // ffmpeg 디코더 초기화
        InitFfmpegDecoder();
        StartStream();
        videoFilePath = Application.dataPath + "/";
        SetVideoFilePath(videoFilePath);
        
        // 런타임 제한 및 메모리 모니터링 코루틴 시작
        StartCoroutine(MonitorRuntime());
        StartCoroutine(MonitorMemoryUsage());
        
        pluginUpdateCoroutine = StartCoroutine(CallPluginAtEndOfFrames());
        yield return null;
    }
    
    private void Update()
    {
        frameCounter++;
        
        // 프레임 간격에 따라 텍스처 업데이트 제한
        if (frameCounter % textureUpdateInterval != 0)
            return;
            
        // 한 프레임당 처리할 최대 패킷 수 제한
        int packetProcessed = 0;
        const int MAX_PACKETS_PER_FRAME = 4;
        
        for (int streamIndex = 0; streamIndex < 2; streamIndex++)
        {
            while (packetProcessed < MAX_PACKETS_PER_FRAME && GetNextRawVideoPacket(streamIndex, out IntPtr ptr, out int size))
            {
                try
                {
                    if (ptr == IntPtr.Zero || size <= 0)
                    {
                        Debug.LogWarning("Invalid video packet received");
                        continue;
                    }
                    
                    byte[] raw = new byte[size];
                    Marshal.Copy(ptr, raw, 0, size);
                    FreeRawVideoPacket(ptr);
                    packetProcessed++;

                    if (streamIndex == 0)
                    {
                        DecodeRawPacket(raw, size, streamIndex, frontTexture);
                    }
                    else if (streamIndex == 1)
                    {
                        DecodeRawPacket(raw, size, streamIndex, backTexture);
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Error processing video packet: {ex.Message}");
                    if (ptr != IntPtr.Zero)
                        FreeRawVideoPacket(ptr);
                }
            }
        }
    }
    
    // 런타임 제한 모니터링
    private IEnumerator MonitorRuntime()
    {
        float startTime = Time.time;
        
        while (true)
        {
            // 최대 실행 시간을 초과하면 리소스 정리 및 재시작
            if ((Time.time - startTime) > maxRuntimeMinutes * 60f)
            {
                Debug.Log("Maximum runtime reached. Cleaning up resources and restarting stream...");
                CleanupResources();
                RestartStream();
                startTime = Time.time; // 타이머 재설정
            }
            
            yield return new WaitForSeconds(30f); // 30초마다 체크
        }
    }
    
    // 메모리 사용량 모니터링
    private IEnumerator MonitorMemoryUsage()
    {
        while (true)
        {
            // 현재 사용 중인 메모리 확인 (MB 단위)
            float memoryUsage = (float)System.GC.GetTotalMemory(false) / (1024f * 1024f);
            
            // 메모리 사용량이 임계값을 초과하면 정리
            if (memoryUsage > MAX_MEMORY_THRESHOLD)
            {
                Debug.LogWarning($"High memory usage detected: {memoryUsage:F2}MB. Cleaning up resources...");
                System.GC.Collect();
                yield return new WaitForSeconds(0.5f);
                
                // 정리 후에도 여전히 메모리가 높으면 스트림 재시작
                memoryUsage = (float)System.GC.GetTotalMemory(true) / (1024f * 1024f);
                if (memoryUsage > MAX_MEMORY_THRESHOLD * 0.8f)
                {
                    Debug.LogWarning("Memory still high after collection. Restarting stream...");
                    RestartStream();
                }
            }
            
            yield return new WaitForSeconds(MEMORY_CHECK_INTERVAL);
        }
    }
    
    // 스트림 재시작 함수
    private void RestartStream()
    {
        StopStream();
        StartCoroutine(DelayedRestart());
    }
    
    private IEnumerator DelayedRestart()
    {
        yield return new WaitForSeconds(1f);
        System.GC.Collect();
        yield return new WaitForSeconds(1f);
        StartStream();
    }
    
    // 리소스 정리 함수
    private void CleanupResources()
    {
        System.GC.Collect();
        System.GC.WaitForPendingFinalizers();
        System.GC.Collect();
    }

    void OnDisable()
    {
        CleanupOnExit();
    }
    
    void OnApplicationQuit()
    {
        CleanupOnExit();
    }
    
    void OnDestroy()
    {
        CleanupOnExit();
    }
    
    // 종료 시 모든 리소스 정리
    private void CleanupOnExit()
    {
        if (pluginUpdateCoroutine != null)
        {
            StopCoroutine(pluginUpdateCoroutine);
            pluginUpdateCoroutine = null;
        }
        
        StopAllCoroutines();
        StopStream();
        StopSaveVideo();
        
        // 가비지 컬렉션 강제 실행
        System.GC.Collect();
        System.GC.WaitForPendingFinalizers();
        System.GC.Collect();
    }

    private void OnPreCull()
    {
        GL.IssuePluginEvent(GetRenderEventFunc(), 0);
    }

    void SetStreamTexture()
    {
        // 이전 텍스처가 있으면 정리
        if (frontTexture != null)
        {
            Destroy(frontTexture);
        }
        if (backTexture != null)
        {
            Destroy(backTexture);
        }
        
        frontTexture = new Texture2D(width, height, TextureFormat.RGBA32, false)
        {
            filterMode = FilterMode.Point
        };
        frontTexture.Apply();

        backTexture = new Texture2D(width, height, TextureFormat.RGBA32, false)
        {
            filterMode = FilterMode.Point
        };
        backTexture.Apply();

        skyboxMaterial.SetTexture("_FrontTex", frontTexture);
        skyboxMaterial.SetTexture("_BackTex", backTexture);
    }

    int updateTimeCounter = 0;

    private IEnumerator CallPluginAtEndOfFrames()
    {
        float startTime = Time.time;
        
        while (true)
        {
            yield return new WaitForEndOfFrame();

            // 애플리케이션이 백그라운드에 있을 때 처리 생략
            if (!Application.isFocused)
            {
                yield return new WaitForSeconds(0.5f);
                continue;
            }

            if (SystemInfo.graphicsDeviceType != GraphicsDeviceType.Direct3D12 &&
                SystemInfo.graphicsDeviceType != GraphicsDeviceType.Switch)
            {
                ++updateTimeCounter;
                SetTimeFromUnity((float)updateTimeCounter * 0.016f);
            }

            if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.Direct3D12)
            {
                GL.IssuePluginEvent(GetRenderEventFunc(), 1);
            }
            
            // 10분마다 updateTimeCounter 리셋
            if (Time.time - startTime > 600f)
            {
                updateTimeCounter = 0;
                startTime = Time.time;
            }
        }
    }

    public bool DecodeRawPacket(byte[] input, int inSize, int streamIndex, [NotNull] Texture2D resultTexture)
    {
        if (resultTexture == null)
        {
            Debug.LogError("[DecodeRawPacket] resultTexture is null!");
            return false;
        }

        IntPtr outData = IntPtr.Zero;
        
        try
        {
            GetDecodedPacket(input, inSize, streamIndex, out outData, out int outputSize);

            if (outputSize > 0 && outData != IntPtr.Zero)
            {
                byte[] decodedBytes = new byte[outputSize];
                Marshal.Copy(outData, decodedBytes, 0, outputSize);
                resultTexture.LoadRawTextureData(decodedBytes);
                resultTexture.Apply(false);
                return true;
            }

            return false;
        }
        catch (Exception ex)
        {
            Debug.LogError($"[DecodeRawPacket] Exception: {ex.Message}");
            return false;
        }
        finally
        {
            // 중요: 할당된 메모리 해제
            if (outData != IntPtr.Zero)
            {
                FreeRawVideoPacket(outData);
            }
        }
    }
}