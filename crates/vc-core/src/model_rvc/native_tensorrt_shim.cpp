#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
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

struct NativeEngine {
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context;
    cudaStream_t stream{};
    std::vector<DeviceBuffer> buffers;
    std::map<std::string, int32_t> tensor_indices;
    std::map<std::string, nvinfer1::Dims> input_dims;
    std::string output_name;
    std::size_t output_len{0};

    ~NativeEngine() {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

Logger& trt_logger() {
    // TensorRT keeps the first logger registered in process-global state.
    // Use one stable instance for every builder/runtime/plugin call; passing
    // stack-local loggers across repeated model builds leaves TensorRT referring
    // to a dead object and also changes behavior compared with trtexec.
    static Logger logger;
    return logger;
}

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

bool cuda_ok(cudaError_t status, Message& msg, char const* what) {
    if (status == cudaSuccess) {
        return true;
    }
    msg.append("%s failed: %s\n", what, cudaGetErrorString(status));
    return false;
}

std::vector<std::string> split(std::string const& value, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        if (!item.empty()) {
            parts.push_back(item);
        }
    }
    return parts;
}

bool parse_dims(std::string const& text, nvinfer1::Dims& dims, Message& msg) {
    auto parts = split(text, 'x');
    if (parts.empty() || parts.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS)) {
        msg.append("invalid TensorRT profile dims: %s\n", text.c_str());
        return false;
    }
    dims.nbDims = static_cast<int32_t>(parts.size());
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        char* end = nullptr;
        long value = std::strtol(parts[static_cast<std::size_t>(i)].c_str(), &end, 10);
        if (end == nullptr || *end != '\0' || value <= 0) {
            msg.append("invalid TensorRT profile dim: %s\n", parts[static_cast<std::size_t>(i)].c_str());
            return false;
        }
        dims.d[i] = static_cast<int64_t>(value);
    }
    return true;
}

bool parse_profile_shapes(char const* profile_shapes, std::map<std::string, nvinfer1::Dims>& shapes, Message& msg) {
    if (profile_shapes == nullptr || profile_shapes[0] == '\0') {
        msg.append("TensorRT profile shape string is empty\n");
        return false;
    }
    for (auto const& item : split(profile_shapes, ',')) {
        auto separator = item.find(':');
        if (separator == std::string::npos || separator == 0 || separator + 1 >= item.size()) {
            msg.append("invalid TensorRT profile entry: %s\n", item.c_str());
            return false;
        }
        nvinfer1::Dims dims{};
        if (!parse_dims(item.substr(separator + 1), dims, msg)) {
            return false;
        }
        shapes[item.substr(0, separator)] = dims;
    }
    return true;
}

bool read_file(char const* path, std::vector<char>& data, Message& msg, char const* label) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        msg.append("failed to open %s: %s\n", label, path);
        return false;
    }
    data.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (data.empty()) {
        msg.append("%s is empty: %s\n", label, path);
        return false;
    }
    return true;
}

int32_t tensor_index(NativeEngine& native, char const* name) {
    auto iter = native.tensor_indices.find(name == nullptr ? "" : name);
    return iter == native.tensor_indices.end() ? -1 : iter->second;
}

bool copy_to_device(NativeEngine& native, char const* name, void const* src, std::size_t bytes, Message& msg) {
    int32_t index = tensor_index(native, name);
    if (index < 0) {
        msg.append("engine is missing tensor %s\n", name);
        return false;
    }
    auto& buffer = native.buffers[static_cast<std::size_t>(index)];
    if (bytes != buffer.bytes) {
        msg.append("TensorRT input %s byte mismatch: got %zu, expected %zu\n", name, bytes, buffer.bytes);
        return false;
    }
    return cuda_ok(cudaMemcpyAsync(buffer.ptr, src, bytes, cudaMemcpyHostToDevice, native.stream), msg, name);
}

bool copy_output_to_host(NativeEngine& native, float* dst, std::size_t output_len, Message& msg) {
    int32_t index = tensor_index(native, native.output_name.c_str());
    if (index < 0) {
        msg.append("engine is missing output tensor %s\n", native.output_name.c_str());
        return false;
    }
    if (output_len != native.output_len) {
        msg.append("TensorRT output length mismatch: got %zu, expected %zu\n", output_len, native.output_len);
        return false;
    }
    return cuda_ok(
        cudaMemcpyAsync(dst, native.buffers[static_cast<std::size_t>(index)].ptr, output_len * sizeof(float), cudaMemcpyDeviceToHost, native.stream),
        msg,
        "cudaMemcpyAsync output"
    );
}

bool enqueue_and_copy(NativeEngine& native, float* output, std::size_t output_len, Message& msg) {
    if (!native.context->enqueueV3(native.stream)) {
        msg.append("TensorRT enqueueV3 failed\n");
        return false;
    }
    if (!copy_output_to_host(native, output, output_len, msg)) {
        return false;
    }
    return cuda_ok(cudaStreamSynchronize(native.stream), msg, "cudaStreamSynchronize after enqueue");
}

} // namespace

extern "C" NativeEngine* vc_rs_trt_engine_create(
    char const* engine_path,
    char const* profile_shapes,
    char const* output_name,
    char* message,
    std::size_t message_len
) {
    Message msg{message, message_len};
    if (message_len > 0) {
        message[0] = '\0';
    }
    if (engine_path == nullptr || profile_shapes == nullptr || output_name == nullptr) {
        msg.append("invalid TensorRT engine create arguments\n");
        return nullptr;
    }
    std::map<std::string, nvinfer1::Dims> profile;
    if (!parse_profile_shapes(profile_shapes, profile, msg)) {
        return nullptr;
    }
    std::vector<char> plan;
    if (!read_file(engine_path, plan, msg, "TensorRT engine")) {
        return nullptr;
    }

    std::unique_ptr<NativeEngine> native(new NativeEngine());
    native->input_dims = profile;
    native->output_name = output_name;
    Logger& logger = trt_logger();
    initLibNvInferPlugins(&logger, "");
    native->runtime.reset(nvinfer1::createInferRuntime(logger));
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
        if (name == nullptr) {
            msg.append("TensorRT engine has null tensor name at index %d\n", i);
            return nullptr;
        }
        native->tensor_indices[name] = i;
        if (native->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            auto iter = profile.find(name);
            if (iter == profile.end()) {
                msg.append("engine input %s is missing from profile %s\n", name, profile_shapes);
                return nullptr;
            }
            auto model_dims = native->engine->getTensorShape(name);
            if (has_dynamic_dim(model_dims) && !native->context->setInputShape(name, iter->second)) {
                msg.append("setInputShape failed for %s wanted=%s\n", name, dims_to_string(iter->second).c_str());
                return nullptr;
            }
            auto actual = native->context->getTensorShape(name);
            if (!same_dims(actual, iter->second)) {
                msg.append("engine input %s has shape %s, expected %s\n", name, dims_to_string(actual).c_str(), dims_to_string(iter->second).c_str());
                return nullptr;
            }
        }
    }

    int32_t output_index = tensor_index(*native, output_name);
    if (output_index < 0) {
        msg.append("engine output %s is missing\n", output_name);
        return nullptr;
    }
    if (native->engine->getTensorDataType(output_name) != nvinfer1::DataType::kFLOAT) {
        msg.append("engine output %s must be FP32\n", output_name);
        return nullptr;
    }
    native->output_len = volume(native->context->getTensorShape(output_name));
    if (native->output_len == 0) {
        msg.append("engine output %s has zero volume\n", output_name);
        return nullptr;
    }

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

    msg.append("loaded native TensorRT engine=%s output=%s output_len=%zu profile=%s\n", engine_path, output_name, native->output_len, profile_shapes);
    return native.release();
}

extern "C" void vc_rs_trt_engine_destroy(NativeEngine* native) {
    delete native;
}

extern "C" std::size_t vc_rs_trt_engine_output_len(NativeEngine const* native) {
    return native == nullptr ? 0 : native->output_len;
}

extern "C" int vc_rs_trt_contentvec_infer(
    NativeEngine* native,
    char const* input_name,
    float const* audio,
    std::size_t audio_len,
    float* output,
    std::size_t output_len,
    char* message,
    std::size_t message_len
) {
    Message msg{message, message_len};
    if (message_len > 0) {
        message[0] = '\0';
    }
    if (native == nullptr || input_name == nullptr || audio == nullptr || output == nullptr) {
        msg.append("null argument passed to TensorRT ContentVec infer\n");
        return 2;
    }
    if (!copy_to_device(*native, input_name, audio, audio_len * sizeof(float), msg)) {
        return 1;
    }
    return enqueue_and_copy(*native, output, output_len, msg) ? 0 : 1;
}

extern "C" int vc_rs_trt_rmvpe_infer(
    NativeEngine* native,
    float const* waveform,
    std::size_t waveform_len,
    float threshold,
    float* output,
    std::size_t output_len,
    char* message,
    std::size_t message_len
) {
    Message msg{message, message_len};
    if (message_len > 0) {
        message[0] = '\0';
    }
    if (native == nullptr || waveform == nullptr || output == nullptr) {
        msg.append("null argument passed to TensorRT RMVPE infer\n");
        return 2;
    }
    if (!copy_to_device(*native, "waveform", waveform, waveform_len * sizeof(float), msg)) {
        return 1;
    }
    if (!copy_to_device(*native, "threshold", &threshold, sizeof(float), msg)) {
        return 1;
    }
    return enqueue_and_copy(*native, output, output_len, msg) ? 0 : 1;
}

extern "C" int vc_rs_trt_rvc_infer(
    NativeEngine* native,
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
    int64_t p_len = static_cast<int64_t>(pitch_len);
    if (!copy_to_device(*native, "feats", feats, feats_len * sizeof(float), msg)) {
        return 1;
    }
    if (!copy_to_device(*native, "p_len", &p_len, sizeof(int64_t), msg)) {
        return 1;
    }
    if (!copy_to_device(*native, "pitch", pitch, pitch_len * sizeof(int64_t), msg)) {
        return 1;
    }
    if (!copy_to_device(*native, "pitchf", pitchf, pitchf_len * sizeof(float), msg)) {
        return 1;
    }
    if (!copy_to_device(*native, "sid", &speaker_id, sizeof(int64_t), msg)) {
        return 1;
    }
    return enqueue_and_copy(*native, output, output_len, msg) ? 0 : 1;
}
