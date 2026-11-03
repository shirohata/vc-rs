#pragma once

#include <filesystem>
#include <system_error>

namespace vc_rs {

// Rust passes UTF-8 paths across the native FFI. The Windows narrow stream
// constructors interpret them using the process code page and retain MAX_PATH.
// Both the builder's writer and the runtime's reader must use the same wide,
// extended-length path, including for temporary engines and timing caches.
inline std::filesystem::path native_file_path(char const* utf8_path) {
    auto path = std::filesystem::u8path(utf8_path);
#ifdef _WIN32
    path.make_preferred();
    auto const original = path.native();
    if (original.rfind(L"\\\\?\\", 0) == 0 || original.rfind(L"\\\\.\\", 0) == 0) {
        return path;
    }
    std::error_code error;
    auto absolute = std::filesystem::absolute(path, error);
    if (error) {
        return path; // Let the stream report the original open failure.
    }
    absolute = absolute.lexically_normal();
    absolute.make_preferred();
    auto const wide = absolute.native();
    if (wide.rfind(L"\\\\", 0) == 0) {
        return std::filesystem::path(L"\\\\?\\UNC\\" + wide.substr(2));
    }
    return std::filesystem::path(L"\\\\?\\" + wide);
#else
    return path;
#endif
}

} // namespace vc_rs
