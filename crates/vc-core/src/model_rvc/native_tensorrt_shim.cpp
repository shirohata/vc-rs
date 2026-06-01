#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, char const* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TRT] %s\n", message);
        }
    }
};

struct Message {
    char* data;
    std::size_t len;
    std::size_t used{0};

    void append(char const* fmt, ...) {
        if (len == 0 || used >= len) {
            return;
        }
        va_list args;
        va_start(args, fmt);
        int written = std::vsnprintf(data + used, len - used, fmt, args);
        va_end(args);
        if (written < 0) {
            return;
        }
        used = std::min(len - 1, used + static_cast<std::size_t>(written));
        data[used] = '\0';
    }
};

template <typename T>
struct TrtDeleter {
    void operator()(T* ptr) const noexcept {
        delete ptr;
    }
};

struct DeviceBuffer {
    void* ptr{nullptr};
    std::size_t bytes{0};

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    bool allocate(std::size_t byte_count, Message& msg, char const* name) {
        bytes = byte_count;
        if (bytes == 0) {
            msg.append("zero-sized TensorRT buffer for %s\n", name);
            return false;
        }
        auto status = cudaMalloc(&ptr, bytes);
        if (status != cudaSuccess) {
            msg.append("cudaMalloc(%s, %zu) failed: %s\n", name, bytes, cudaGetErrorString(status));
            return false;
        }
        return true;
    }
};

struct TensorSlot {
    int32_t index{-1};
    std::string name;
    nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
    nvinfer1::Dims dims{};
};

struct NativeRvcEngine {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context;
    cudaStream_t stream{};
    TensorSlot feats;
    TensorSlot p_len;
    TensorSlot pitch;
    TensorSlot pitchf;
    TensorSlot sid;
    TensorSlot audio;
    std::vector<DeviceBuffer> buffers;
    int32_t frames{0};
    int32_t channels{0};
    std::size_t output_len{0};

    ~NativeRvcEngine() {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

std::string dims_to_string(nvinfer1::Dims const& dims) {
    if (dims.nbDims < 0) {
        return "<invalid>";
    }
    std::ostringstream out;
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        if (i != 0) {
            out << 'x';
        }
        out << dims.d[i];
    }
    return out.str();
}

std::size_t dtype_size(nvinfer1::DataType dtype) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
        return 2;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
        return 1;
    case nvinfer1::DataType::kINT64:
        return 8;
    default:
        return 0;
    }
}

std::size_t volume(nvinfer1::Dims const& dims) {
    if (dims.nbDims < 0) {
        return 0;
    }
    std::size_t v = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return 0;
        }
        v *= static_cast<std::size_t>(dims.d[i]);
    }
    return v;
}

bool cuda_ok(cudaError_t status, Message& msg, char const* what) {
    if (status == cudaSuccess) {
        return true;
    }
    msg.append("%s failed: %s\n", what, cudaGetErrorString(status));
    return false;
}

bool has_dynamic_dim(nvinfer1::Dims const& dims) {
    if (dims.nbDims < 0) {
        return true;
    }
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return true;
        }
    }
    return false;
}

nvinfer1::Dims expected_input_dims(char const* name, nvinfer1::Dims dims, int32_t frames, int32_t channels) {
    std::string tensor(name);
    if (tensor == "feats") {
        dims.nbDims = 3;
        dims.d[0] = 1;
        dims.d[1] = frames;
        dims.d[2] = channels;
    } else if (tensor == "pitch" || tensor == "pitchf") {
        dims.nbDims = 2;
        dims.d[0] = 1;
        dims.d[1] = frames;
    } else if (tensor == "p_len" || tensor == "sid") {
        dims.nbDims = 1;
        dims.d[0] = 1;
    }
    return dims;
}

bool same_dims(nvinfer1::Dims const& a, nvinfer1::Dims const& b) {
    if (a.nbDims != b.nbDims) {
        return false;
    }
    for (int32_t i = 0; i < a.nbDims; ++i) {
        if (a.d[i] != b.d[i]) {
            return false;
        }
    }
    return true;
}

TensorSlot* slot_for_name(NativeRvcEngine& native, char const* name) {
    std::string tensor(name);
    if (tensor == "feats") {
        return &native.feats;
    }
    if (tensor == "p_len") {
        return &native.p_len;
    }
    if (tensor == "pitch") {
        return &native.pitch;
    }
    if (tensor == "pitchf") {
        return &native.pitchf;
    }
    if (tensor == "sid") {
        return &native.sid;
    }
    if (tensor == "audio") {
        return &native.audio;
    }
    return nullptr;
}

bool validate_slot(TensorSlot const& slot, nvinfer1::DataType dtype, nvinfer1::Dims const& dims, Message& msg) {
    if (slot.index < 0) {
        msg.append("engine is missing tensor %s\n", slot.name.c_str());
        return false;
    }
    if (slot.dtype != dtype) {
        msg.append("tensor %s has unexpected dtype\n", slot.name.c_str());
        return false;
    }
    if (!same_dims(slot.dims, dims)) {
        msg.append("tensor %s has shape %s, expected %s\n", slot.name.c_str(), dims_to_string(slot.dims).c_str(), dims_to_string(dims).c_str());
        return false;
    }
    return true;
}

bool copy_to_device(void* dst, void const* src, std::size_t bytes, cudaStream_t stream, Message& msg, char const* name) {
    return cuda_ok(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream), msg, name);
}

bool copy_output_to_host(void* dst, void const* src, std::size_t bytes, cudaStream_t stream, Message& msg) {
    return cuda_ok(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream), msg, "cudaMemcpyAsync output");
}

bool run_inference(
    NativeRvcEngine& native,
    float const* feats,
    std::size_t feats_len,
    int64_t const* pitch,
    std::size_t pitch_len,
    float const* pitchf,
    std::size_t pitchf_len,
    int64_t speaker_id,
    float* output,
    std::size_t output_len,
    Message& msg
) {
    std::size_t expected_feats = static_cast<std::size_t>(native.frames) * static_cast<std::size_t>(native.channels);
    std::size_t expected_pitch = static_cast<std::size_t>(native.frames);
    if (feats_len != expected_feats || pitch_len != expected_pitch || pitchf_len != expected_pitch || output_len != native.output_len) {
        msg.append("TensorRT RVC shape mismatch feats=%zu/%zu pitch=%zu/%zu pitchf=%zu/%zu output=%zu/%zu\n",
            feats_len,
            expected_feats,
            pitch_len,
            expected_pitch,
            pitchf_len,
            expected_pitch,
            output_len,
            native.output_len);
        return false;
    }

    int64_t p_len = native.frames;
    if (!copy_to_device(native.buffers[static_cast<std::size_t>(native.feats.index)].ptr, feats, feats_len * sizeof(float), native.stream, msg, "copy feats")) {
        return false;
    }
    if (!copy_to_device(native.buffers[static_cast<std::size_t>(native.p_len.index)].ptr, &p_len, sizeof(int64_t), native.stream, msg, "copy p_len")) {
        return false;
    }
    if (!copy_to_device(native.buffers[static_cast<std::size_t>(native.pitch.index)].ptr, pitch, pitch_len * sizeof(int64_t), native.stream, msg, "copy pitch")) {
        return false;
    }
    if (!copy_to_device(native.buffers[static_cast<std::size_t>(native.pitchf.index)].ptr, pitchf, pitchf_len * sizeof(float), native.stream, msg, "copy pitchf")) {
        return false;
    }
    if (!copy_to_device(native.buffers[static_cast<std::size_t>(native.sid.index)].ptr, &speaker_id, sizeof(int64_t), native.stream, msg, "copy sid")) {
        return false;
    }
    if (!native.context->enqueueV3(native.stream)) {
        msg.append("TensorRT RVC enqueueV3 failed\n");
        return false;
    }
    if (!copy_output_to_host(output, native.buffers[static_cast<std::size_t>(native.audio.index)].ptr, output_len * sizeof(float), native.stream, msg)) {
        return false;
    }
    return cuda_ok(cudaStreamSynchronize(native.stream), msg, "cudaStreamSynchronize after enqueue");
}

} // namespace

extern "C" NativeRvcEngine* vc_rs_trt_rvc_create(
    char const* engine_path,
    int32_t frames,
    int32_t channels,
    char* message,
    std::size_t message_len
) {
    Message msg{message, message_len};
    if (message_len > 0) {
        message[0] = '\0';
    }
    if (engine_path == nullptr || frames <= 0 || channels <= 0) {
        msg.append("invalid TensorRT RVC create arguments\n");
        return nullptr;
    }

    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        msg.append("failed to open TensorRT RVC engine: %s\n", engine_path);
        return nullptr;
    }
    std::vector<char> plan((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (plan.empty()) {
        msg.append("TensorRT RVC engine is empty: %s\n", engine_path);
        return nullptr;
    }

    std::unique_ptr<NativeRvcEngine> native(new NativeRvcEngine());
    native->frames = frames;
    native->channels = channels;
    initLibNvInferPlugins(&native->logger, "");

    native->runtime.reset(nvinfer1::createInferRuntime(native->logger));
    if (!native->runtime) {
        msg.append("createInferRuntime failed\n");
        return nullptr;
    }
    native->engine.reset(native->runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!native->engine) {
        msg.append("deserializeCudaEngine failed for %s\n", engine_path);
        return nullptr;
    }
    native->context.reset(native->engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kSTATIC));
    if (!native->context) {
        msg.append("createExecutionContext failed\n");
        return nullptr;
    }
    if (!cuda_ok(cudaStreamCreate(&native->stream), msg, "cudaStreamCreate")) {
        return nullptr;
    }

    int32_t const nb_io = native->engine->getNbIOTensors();
    native->buffers.resize(static_cast<std::size_t>(nb_io));
    for (int32_t i = 0; i < nb_io; ++i) {
        char const* name = native->engine->getIOTensorName(i);
        auto* slot = slot_for_name(*native, name);
        if (slot == nullptr) {
            continue;
        }
        slot->index = i;
        slot->name = name;
        slot->dtype = native->engine->getTensorDataType(name);
        slot->dims = native->engine->getTensorShape(name);
        if (native->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            auto wanted = expected_input_dims(name, slot->dims, frames, channels);
            if (has_dynamic_dim(slot->dims) && !native->context->setInputShape(name, wanted)) {
                msg.append("setInputShape failed for %s wanted=%s\n", name, dims_to_string(wanted).c_str());
                return nullptr;
            }
        }
        slot->dims = native->context->getTensorShape(name);
    }

    if (!validate_slot(native->feats, nvinfer1::DataType::kFLOAT, expected_input_dims("feats", native->feats.dims, frames, channels), msg)
        || !validate_slot(native->p_len, nvinfer1::DataType::kINT64, expected_input_dims("p_len", native->p_len.dims, frames, channels), msg)
        || !validate_slot(native->pitch, nvinfer1::DataType::kINT64, expected_input_dims("pitch", native->pitch.dims, frames, channels), msg)
        || !validate_slot(native->pitchf, nvinfer1::DataType::kFLOAT, expected_input_dims("pitchf", native->pitchf.dims, frames, channels), msg)
        || !validate_slot(native->sid, nvinfer1::DataType::kINT64, expected_input_dims("sid", native->sid.dims, frames, channels), msg)
        || !validate_slot(native->audio, nvinfer1::DataType::kFLOAT, native->context->getTensorShape("audio"), msg)) {
        return nullptr;
    }

    native->output_len = volume(native->audio.dims);
    for (int32_t i = 0; i < nb_io; ++i) {
        char const* name = native->engine->getIOTensorName(i);
        auto dims = native->context->getTensorShape(name);
        auto dtype = native->engine->getTensorDataType(name);
        std::size_t bytes = volume(dims) * dtype_size(dtype);
        if (!native->buffers[static_cast<std::size_t>(i)].allocate(bytes, msg, name)) {
            return nullptr;
        }
        if (!native->context->setTensorAddress(name, native->buffers[static_cast<std::size_t>(i)].ptr)) {
            msg.append("setTensorAddress failed for %s\n", name);
            return nullptr;
        }
    }

    msg.append("loaded native TensorRT RVC engine=%s frames=%d channels=%d output=%zu\n", engine_path, frames, channels, native->output_len);
    return native.release();
}

extern "C" void vc_rs_trt_rvc_destroy(NativeRvcEngine* native) {
    delete native;
}

extern "C" std::size_t vc_rs_trt_rvc_output_len(NativeRvcEngine const* native) {
    return native == nullptr ? 0 : native->output_len;
}

extern "C" int vc_rs_trt_rvc_infer(
    NativeRvcEngine* native,
    float const* feats,
    std::size_t feats_len,
    int64_t const* pitch,
    std::size_t pitch_len,
    float const* pitchf,
    std::size_t pitchf_len,
    int64_t speaker_id,
    float* output,
    std::size_t output_len,
    char* message,
    std::size_t message_len
) {
    Message msg{message, message_len};
    if (message_len > 0) {
        message[0] = '\0';
    }
    if (native == nullptr || feats == nullptr || pitch == nullptr || pitchf == nullptr || output == nullptr) {
        msg.append("null argument passed to TensorRT RVC infer\n");
        return 2;
    }
    return run_inference(*native, feats, feats_len, pitch, pitch_len, pitchf, pitchf_len, speaker_id, output, output_len, msg) ? 0 : 1;
}
