#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

class ProbeLogger final : public nvinfer1::ILogger {
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

char const* dtype_name(nvinfer1::DataType dtype) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
        return "float32";
    case nvinfer1::DataType::kHALF:
        return "float16";
    case nvinfer1::DataType::kINT8:
        return "int8";
    case nvinfer1::DataType::kINT32:
        return "int32";
    case nvinfer1::DataType::kBOOL:
        return "bool";
    case nvinfer1::DataType::kUINT8:
        return "uint8";
    case nvinfer1::DataType::kBF16:
        return "bf16";
    case nvinfer1::DataType::kINT64:
        return "int64";
    default:
        return "other";
    }
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

nvinfer1::Dims input_dims_for(char const* name, nvinfer1::Dims dims, int32_t frames, int32_t channels) {
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
    }
    return dims;
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

bool upload_dummy_input(
    char const* name,
    nvinfer1::DataType dtype,
    nvinfer1::Dims const& dims,
    DeviceBuffer& buffer,
    cudaStream_t stream,
    int32_t frames,
    Message& msg
) {
    std::size_t elems = volume(dims);
    std::size_t elem_size = dtype_size(dtype);
    if (elems == 0 || elem_size == 0) {
        msg.append("unsupported or unresolved input %s dtype=%s shape=%s\n", name, dtype_name(dtype), dims_to_string(dims).c_str());
        return false;
    }
    buffer.bytes = elems * elem_size;
    if (!cuda_ok(cudaMalloc(&buffer.ptr, buffer.bytes), msg, "cudaMalloc input")) {
        return false;
    }

    std::string tensor(name);
    if (dtype == nvinfer1::DataType::kINT64) {
        std::vector<int64_t> host(elems, 0);
        if (tensor == "p_len") {
            std::fill(host.begin(), host.end(), static_cast<int64_t>(frames));
        } else if (tensor == "pitch") {
            std::fill(host.begin(), host.end(), static_cast<int64_t>(1));
        }
        return cuda_ok(cudaMemcpyAsync(buffer.ptr, host.data(), buffer.bytes, cudaMemcpyHostToDevice, stream), msg, "cudaMemcpyAsync input int64");
    }

    return cuda_ok(cudaMemsetAsync(buffer.ptr, 0, buffer.bytes, stream), msg, "cudaMemsetAsync input");
}

} // namespace

extern "C" int trt_probe_engine(
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
    if (engine_path == nullptr) {
        msg.append("engine path is null\n");
        return 2;
    }
    if (frames <= 0 || channels <= 0) {
        msg.append("frames/channels must be positive\n");
        return 2;
    }

    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        msg.append("failed to open engine: %s\n", engine_path);
        return 2;
    }
    std::vector<char> plan((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (plan.empty()) {
        msg.append("engine file is empty: %s\n", engine_path);
        return 2;
    }

    ProbeLogger logger;
    initLibNvInferPlugins(&logger, "");

    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        msg.append("createInferRuntime failed\n");
        return 1;
    }

    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine(
        runtime->deserializeCudaEngine(plan.data(), plan.size())
    );
    if (!engine) {
        msg.append("deserializeCudaEngine failed\n");
        return 1;
    }

    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context(
        engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kSTATIC)
    );
    if (!context) {
        msg.append("createExecutionContext failed\n");
        return 1;
    }

    cudaStream_t stream{};
    if (!cuda_ok(cudaStreamCreate(&stream), msg, "cudaStreamCreate")) {
        return 1;
    }

    int32_t const nb_io = engine->getNbIOTensors();
    std::vector<DeviceBuffer> buffers(static_cast<std::size_t>(nb_io));

    msg.append("loaded engine: %s\n", engine_path);
    msg.append("io tensors: %d\n", nb_io);

    for (int32_t i = 0; i < nb_io; ++i) {
        char const* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        auto dtype = engine->getTensorDataType(name);
        auto dims = engine->getTensorShape(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            nvinfer1::Dims wanted = input_dims_for(name, dims, frames, channels);
            if (has_dynamic_dim(dims) && !context->setInputShape(name, wanted)) {
                msg.append("setInputShape failed for %s wanted=%s\n", name, dims_to_string(wanted).c_str());
                cudaStreamDestroy(stream);
                return 1;
            }
        }
        msg.append("  [%d] %s %s %s engine_shape=%s\n", i, mode == nvinfer1::TensorIOMode::kINPUT ? "input" : "output", name, dtype_name(dtype), dims_to_string(dims).c_str());
    }

    for (int32_t i = 0; i < nb_io; ++i) {
        char const* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        auto dtype = engine->getTensorDataType(name);
        auto dims = context->getTensorShape(name);
        if (volume(dims) == 0 || dtype_size(dtype) == 0) {
            msg.append("unresolved tensor %s runtime_shape=%s dtype=%s\n", name, dims_to_string(dims).c_str(), dtype_name(dtype));
            cudaStreamDestroy(stream);
            return 1;
        }

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            if (!upload_dummy_input(name, dtype, dims, buffers[static_cast<std::size_t>(i)], stream, frames, msg)) {
                cudaStreamDestroy(stream);
                return 1;
            }
        } else {
            buffers[static_cast<std::size_t>(i)].bytes = volume(dims) * dtype_size(dtype);
            if (!cuda_ok(cudaMalloc(&buffers[static_cast<std::size_t>(i)].ptr, buffers[static_cast<std::size_t>(i)].bytes), msg, "cudaMalloc output")) {
                cudaStreamDestroy(stream);
                return 1;
            }
        }

        if (!context->setTensorAddress(name, buffers[static_cast<std::size_t>(i)].ptr)) {
            msg.append("setTensorAddress failed for %s\n", name);
            cudaStreamDestroy(stream);
            return 1;
        }
        msg.append("  bound %s runtime_shape=%s bytes=%zu\n", name, dims_to_string(dims).c_str(), buffers[static_cast<std::size_t>(i)].bytes);
    }

    if (!cuda_ok(cudaStreamSynchronize(stream), msg, "cudaStreamSynchronize before enqueue")) {
        cudaStreamDestroy(stream);
        return 1;
    }
    if (!context->enqueueV3(stream)) {
        msg.append("enqueueV3 failed\n");
        cudaStreamDestroy(stream);
        return 1;
    }
    if (!cuda_ok(cudaStreamSynchronize(stream), msg, "cudaStreamSynchronize after enqueue")) {
        cudaStreamDestroy(stream);
        return 1;
    }

    for (int32_t i = 0; i < nb_io; ++i) {
        char const* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        auto dtype = engine->getTensorDataType(name);
        if (mode != nvinfer1::TensorIOMode::kOUTPUT || dtype != nvinfer1::DataType::kFLOAT) {
            continue;
        }
        auto dims = context->getTensorShape(name);
        std::size_t elems = volume(dims);
        std::vector<float> host(elems);
        if (!cuda_ok(cudaMemcpy(host.data(), buffers[static_cast<std::size_t>(i)].ptr, buffers[static_cast<std::size_t>(i)].bytes, cudaMemcpyDeviceToHost), msg, "cudaMemcpy output")) {
            cudaStreamDestroy(stream);
            return 1;
        }
        double sum_sq = 0.0;
        float min_value = host.empty() ? 0.0F : host[0];
        float max_value = host.empty() ? 0.0F : host[0];
        for (float value : host) {
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            sum_sq += static_cast<double>(value) * static_cast<double>(value);
        }
        double rms = host.empty() ? 0.0 : std::sqrt(sum_sq / static_cast<double>(host.size()));
        msg.append("  output %s samples=%zu min=%.8f max=%.8f rms=%.8f first=%.8f\n",
            name,
            host.size(),
            static_cast<double>(min_value),
            static_cast<double>(max_value),
            rms,
            host.empty() ? 0.0 : static_cast<double>(host[0]));
    }

    cudaStreamDestroy(stream);
    msg.append("enqueueV3 succeeded\n");
    return 0;
}
