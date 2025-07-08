#include "Unity/IUnityGraphics.h"

#include <iostream>

// 카메라 및 디바이스 검색 관련 헤더
#include "camera/camera.h"
#include "camera/device_discovery.h"

// ffmpeg 헤더
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
}

#include "RenderingPluginInterface.h"

#include <array>
#include <mutex>
#include <queue>

static std::mutex g_ffmpegDecodeMutex;

static std::mutex g_codecContextMutex;
static std::array<AVCodecContext*, 2> g_AVCodecContexts = { nullptr, nullptr };

// 카메라 및 스트림 델리게이트 관련 전역 변수들
static std::shared_ptr<ins_camera::Camera> g_Camera = nullptr;
static std::shared_ptr<ins_camera::StreamDelegate> g_Delegate = nullptr;

static AVBufferRef* hw_device_ctx = NULL;
static enum AVPixelFormat hw_pix_fmt;

static int ffmpeg_hw_decoder_init(AVCodecContext* ctx, const enum AVHWDeviceType type)
{
    int err = 0;

    // hw_device_ctx가 이미 생성되어있으면 동작 안하기 때문에 재사용 가능.
    if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type,
        NULL, NULL, 0)) < 0) {
        std::cerr << "[insta360][error] Failed to create HW device context.\n";
        return err;
    }
    ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    return err;
}

static enum AVPixelFormat get_hw_format(AVCodecContext* ctx,
    const enum AVPixelFormat* pix_fmts)
{
    const enum AVPixelFormat* p;

    for (p = pix_fmts; *p != -1; p++) {
        if (*p == hw_pix_fmt)
            return *p;
    }

    std::cerr << "[insta360][error] Failed to get HW surface format.\n";
    return AV_PIX_FMT_NONE;
}

// 두 스트림에 대해 codec context를 초기화하는 함수
int ffmpeg_decoder_init() {
    enum AVHWDeviceType device_type = av_hwdevice_find_type_by_name("cuda");
    if (device_type == AV_HWDEVICE_TYPE_NONE) {
        std::cerr << "[insta360][error] CUDA device type not supported." << std::endl;
        return -1;
    }

    for (int i = 0; i < 2; i++) {
        const AVCodec* decoder = avcodec_find_decoder_by_name("h264_cuvid");
        if (!decoder) {
            std::cerr << "[insta360][error] H.264 (cuvid) decoder not found." << std::endl;
            return -1;
        }

        AVCodecContext* ctx = avcodec_alloc_context3(decoder);
        if (!ctx) {
            std::cerr << "[insta360][error] Failed to allocate codec context for stream " << i << std::endl;
            return AVERROR(ENOMEM);
        }

        // get_format 콜백 등록
        ctx->get_format = get_hw_format;

        const AVCodecHWConfig* config = nullptr;
        for (int j = 0;; j++) {
            config = avcodec_get_hw_config(decoder, j);
            if (!config) {
                std::cerr << "[insta360][error] Decoder " << decoder->name << " does not support CUDA." << std::endl;
                return -1;
            }
            if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
                config->device_type == device_type) {
                hw_pix_fmt = config->pix_fmt;
                break;
            }
        }

        // 하드웨어 디코더 초기화 (hw_device_ctx 할당)
        if (ffmpeg_hw_decoder_init(ctx, device_type) < 0) {
            std::cerr << "[insta360][error] Failed to initialize hardware decoder for stream " << i << std::endl;
            return -1;
        }

        if (avcodec_open2(ctx, decoder, NULL) < 0) {
            std::cerr << "[insta360][error] Failed to open codec for stream " << i << std::endl;
            return -1;
        }

        {
            std::lock_guard<std::mutex> lock(g_codecContextMutex);
            g_AVCodecContexts[i] = ctx;
        }
    }
    return 0;
}

void SaveBGRDataToBMP(const Npp8u* bgrData, int width, int height, int stride, const char* filename, int idx)
{
    // BMP 파일 헤더 크기
    const int headerSize = 54;
    // 파일 크기 계산
    int rowSize = ((width * 3 + 3) & ~3); // 각 행은 4바이트 경계에 맞춰야 함
    int fileSize = headerSize + rowSize * height;

    // 임시 버퍼 할당
    unsigned char* bmpData = (unsigned char*)malloc(fileSize);
    if (!bmpData) {
        std::cerr << "메모리 할당 실패" << "\n";
        return;
    }

    // BMP 헤더 설정
    memset(bmpData, 0, headerSize);
    // BMP 파일 헤더 (14 bytes)
    bmpData[0] = 'B';
    bmpData[1] = 'M';
    *(int*)&bmpData[2] = fileSize;  // 파일 크기
    *(int*)&bmpData[10] = headerSize; // 픽셀 데이터 시작 오프셋

    // DIB 헤더 (40 bytes)
    *(int*)&bmpData[14] = 40;       // DIB 헤더 크기
    *(int*)&bmpData[18] = width;    // 너비
    *(int*)&bmpData[22] = -height;  // 높이 (음수: 상단-좌측부터 시작)
    *(short*)&bmpData[26] = 1;      // 색상 면 수
    *(short*)&bmpData[28] = 24;     // 비트 깊이
    *(int*)&bmpData[34] = 0;        // 압축 없음

    // GPU에서 CPU로 BGR 데이터 복사
    unsigned char* hostBGR = (unsigned char*)malloc(height * stride);
    if (!hostBGR) {
        std::cerr << "BGR 데이터용 메모리 할당 실패" << "\n";
        free(bmpData);
        return;
    }

    cudaError_t status = cudaMemcpy(hostBGR, bgrData, height * stride, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::cerr << "CUDA 메모리 복사 실패: " << cudaGetErrorString(status) << "\n";
        free(hostBGR);
        free(bmpData);
        return;
    }

    // BGR 데이터를 BMP 형식으로 변환
    for (int y = 0; y < height; y++) {
        const unsigned char* srcRow = hostBGR + y * stride;
        unsigned char* dstRow = bmpData + headerSize + y * rowSize;

        for (int x = 0; x < width; x++) {
            dstRow[x * 3 + 0] = srcRow[x * 3 + 0]; // B
            dstRow[x * 3 + 1] = srcRow[x * 3 + 1]; // G
            dstRow[x * 3 + 2] = srcRow[x * 3 + 2]; // R
        }
    }
    std::string extension = "_" + std::to_string(idx) + ".bmp";
    std::string finalPath = filename + extension;
    // 파일에 저장
    FILE* file = fopen(finalPath.c_str(), "wb");
    if (file) {
        fwrite(bmpData, 1, fileSize, file);
        fclose(file);
        //std::cout << "BGR 데이터가 " << finalPath << " 파일로 저장됨" << "\n";
    }
    else {
        std::cerr << "파일 생성 실패: " << finalPath << "\n";
    }

    free(hostBGR);
    free(bmpData);
}

int ffmpeg_decode_packet(const uint8_t* data, size_t size, int stream_index) 
{
    std::lock_guard<std::mutex> lock(g_codecContextMutex);
    if (stream_index < 0 || stream_index >= 2 || !g_AVCodecContexts[stream_index]) {
        std::cerr << "[insta360][error] Invalid stream index: " << stream_index << "\n";
        return -1;
    }

    AVCodecContext* codecCtx = g_AVCodecContexts[stream_index];
    //codecCtx->hw_device_ctx = av_buffer_ref(g_HwDeviceCtxs[stream_index]);

    AVPacket* packet = av_packet_alloc();
    if (!packet) return AVERROR(ENOMEM);
    packet->data = const_cast<uint8_t*>(data);
    packet->size = static_cast<int>(size);

    int ret = avcodec_send_packet(codecCtx, packet);
    av_packet_free(&packet);
    if (ret < 0) {
        std::cerr << "[insta360][error] Error sending packet for decoding.\n";
        return ret;
    }

    // 디코딩된 프레임을 반복적으로 받아옴
    while (ret >= 0) {
        AVFrame* frame = av_frame_alloc();
        if (!frame) {
            std::cerr << "[insta360][error] Failed to allocate frame.\n";
            ret = AVERROR(ENOMEM);
            break;
        }
        ret = avcodec_receive_frame(codecCtx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_frame_free(&frame);
            break;
        }
        else if (ret < 0) {
            std::cerr << "[insta360][error] Error during decoding.\n";
            av_frame_free(&frame);
            break;
        }

        // 하드웨어 픽셀 포맷이면
        if (frame->format == hw_pix_fmt) {
            int stepBytes;
            Npp8u* bgrData = nullptr;
            // Allocate memory for BGR data on the device
            bgrData = nppiMalloc_8u_C3(frame->width, frame->height, &stepBytes);
            if (!bgrData)
            {
                std::cerr << "[insta360][error] nppiMalloc_8u_C3 failed to allocate memory." << "\n";
                return -1;
            }

            NppiSize oSizeROI = { frame->width,frame->height };

            //DBUG_PRINT("width: %d, height: %d, stepBytes: %d\n", oSizeROI.width, oSizeROI.height, stepBytes);
            // NppStatus stat = nppiNV12ToBGR_8u_P2C3R(frame_dec->data, frame_dec->width, bgrData, frame_dec->width*3*sizeof(Npp8u), oSizeROI);
#ifdef USE_709frame->data
            NppStatus stat = nppiNV12ToBGR_8u_P2C3R(frame->data, frame->linesize[0], bgrData, frame->width * 3, oSizeROI);
#else
            NppStatus stat = nppiNV12ToRGB_709HDTV_8u_P2C3R(frame->data, frame->linesize[0], bgrData, frame->width * 3, oSizeROI);
#endif
            //SaveBGRDataToBMP(bgrData, frame->width, frame->height, frame->width * 3, g_Delegate->GetSavePath(), stream_index);
            QueueBGRData(bgrData, frame->width, frame->height, frame->width * 3, stream_index);
        }
        else
        {
            std::cout << "[insta360][warning] hw format is not NV12 format. It is " << frame->format << "\n";
        }

        av_frame_free(&frame);
    }
    return ret;
}


// =====================================================
// 카메라 스트림 및 파일 저장 관련 함수들
// =====================================================
class TestStreamDelegate : public ins_camera::StreamDelegate
{
private:
    FILE* videoFile1_ = nullptr;
    FILE* videoFile2_ = nullptr;
    char videoFilePath1_[512] = { 0 };
    char videoFilePath2_[512] = { 0 };

    bool bCanSaveVideo = false;
        
    void SaveVideoFile(const uint8_t* data, size_t size, int64_t timestamp,
        uint8_t streamType, int stream_index = 0)
    {
        if (!videoFile1_ && videoFilePath1_[0] != '\0')
        {
            if (fopen_s(&videoFile1_, videoFilePath1_, "wb") != 0)
            {
                std::cerr << "[insta360][error] 비디오 파일 1 열기 실패."
                    << "\n";
                return;
            }
        }
        if (stream_index == 0 && videoFile1_)
        {
            fwrite(data, sizeof(uint8_t), size, videoFile1_);
            std::cout << "[Insta360][debug] Video File 1 has been saved.\n";
        }

        if (!videoFile2_ && videoFilePath2_[0] != '\0')
        {
            if (fopen_s(&videoFile2_, videoFilePath2_, "wb") != 0)
            {
                std::cerr << "[insta360][error] 비디오 파일 2 열기 실패."
                    << "\n";
                return;
            }
        }
        if (stream_index == 1 && videoFile2_)
        {
            fwrite(data, sizeof(uint8_t), size, videoFile2_);
            std::cout << "[Insta360][debug] Video File 2 has been saved.\n";
        }
    }

public:
    TestStreamDelegate()
    {
    }
    ~TestStreamDelegate()
    {
        if (videoFile1_)
            fclose(videoFile1_);
        if (videoFile2_)
            fclose(videoFile2_);
    }

    void OnAudioData(const uint8_t* data, size_t size,
        int64_t timestamp) override
    {
    }

    void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp,
        uint8_t streamType, int stream_index = 0) override
    {
        if(bCanSaveVideo)
            SaveVideoFile(data, size, timestamp, streamType, stream_index);
        //std::cout << "[insta360][debug] OnVideoData Callback ---------------" << "\n";
        //std::cout << "  Timestamp: " << timestamp << " ms" << "\n";
        //std::cout << "  Stream Type: " << static_cast<int>(streamType) << "\n";
        //std::cout << "  Stream Index: " << stream_index << "\n";
        //std::cout << "  Data Size: " << size << " bytes" << "\n";
  
        // try_lock으로 락을 획득 시도 (이미 락이 걸려 있다면 바로 리턴)
        std::unique_lock<std::mutex> lock(g_ffmpegDecodeMutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            std::cerr << "[insta360][warning] ffmpeg_decode_packet이 바쁨 - 프레임 무시됨\n";
            return;
        }
        
        ffmpeg_decode_packet(data, size, stream_index);
    }

    void OnGyroData(const std::vector<ins_camera::GyroData>& data) override
    {
    }

    void OnExposureData(const ins_camera::ExposureData& data) override
    {
    }

    void SetVideoPath(const std::string& video_path) override
    {
        const std::string extension = ".h264";
        const std::string videoName1 = "01";
        const std::string videoName2 = "02";
        const std::string path1 = video_path + videoName1 + extension;
        const std::string path2 = video_path + videoName2 + extension;
        strncpy_s(videoFilePath1_, sizeof(videoFilePath1_), path1.c_str(),
            _TRUNCATE);
        strncpy_s(videoFilePath2_, sizeof(videoFilePath2_), path2.c_str(),
            _TRUNCATE);
    }

    void StartSaveVideo()
    {
        bCanSaveVideo = true;
    }

    void StopSaveVideo()
    {
        bCanSaveVideo = false;
    }

    char* GetSavePath()
    {
        return videoFilePath1_;
    }
};

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API StartStream()
{
    if (ffmpeg_decoder_init() < 0)
    {
        std::cerr << "[insta360][error] ffmpeg 하드웨어 디코더 초기화 실패."
            << "\n";
        return;
    }

    ins_camera::DeviceDiscovery discovery;
    auto deviceList = discovery.GetAvailableDevices();
    if (deviceList.empty())
    {
        std::cerr << "[insta360][error] 카메라 장치 없음." << "\n";
        return;
    }

    g_Camera = std::make_shared<ins_camera::Camera>(deviceList[0].info);
    if (!g_Camera->Open())
    {
        std::cerr << "[insta360][error] 카메라 열기 실패." << "\n";
        return;
    }

    g_Delegate = std::make_shared<TestStreamDelegate>();
    g_Camera->SetStreamDelegate(g_Delegate);
    discovery.FreeDeviceDescriptors(deviceList);

    ins_camera::LiveStreamParam param;
    param.video_resolution = ins_camera::VideoResolution::RES_2880_2880P30;
    param.video_bitrate = 1024 * 1024 * 80; // 80 Mbps
    param.enable_audio = false;
    param.using_lrv = false;

    if (!g_Camera->StartLiveStreaming(param))
    {
        std::cerr << "[insta360][error] 라이브 스트림 시작 실패." << "\n";
    }
    else
    {
        std::cout << "[insta360][info] 라이브 스트림 시작 성공." << "\n";
    }
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API StopStream()
{
    if (g_Camera)
    {
        if (g_Camera->StopLiveStreaming())
        {
            std::cout << "[insta360][info] 라이브 스트림 중지 성공."
                << "\n";
        }
        else
        {
            std::cout << "[insta360][error] 라이브 스트림 중지 실패."
                << "\n";
        }
        g_Camera->Close();
        g_Camera.reset();
    }

    // ffmpeg 디코더 context 해제
    {
        std::lock_guard<std::mutex> lock(g_codecContextMutex);
        for (int i = 0; i < g_AVCodecContexts.size(); i++) {
            if (g_AVCodecContexts[i]) {
                avcodec_free_context(&g_AVCodecContexts[i]);
                g_AVCodecContexts[i] = nullptr;
            }
        }
    }

    // 하드웨어 디바이스 컨텍스트 해제
    if (hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
        hw_device_ctx = nullptr;
    }
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetVideoFilePath(const char* path)
{
    if (!g_Delegate)
    {
        std::cerr << "[insta360][error] Delegate가 초기화되지 않음."
            << "\n";
        return;
    }
    g_Delegate->SetVideoPath(path);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API StartSaveVideo()
{
    if (!g_Delegate)
    {
        std::cerr << "[insta360][error] Delegate가 초기화되지 않음."
            << "\n";
        return;
    }
    g_Delegate->StartSaveVideo();
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API StopSaveVideo()
{
    if (!g_Delegate)
    {
        std::cerr << "[insta360][error] Delegate가 초기화되지 않음."
            << "\n";
        return;
    }
    g_Delegate->StopSaveVideo();
}