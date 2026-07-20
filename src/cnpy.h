#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <vector>

namespace cnpy {

struct NpyArray {
    std::vector<size_t> shape;
    std::vector<char> bytes;

    template <typename T>
    T* data() {
        return reinterpret_cast<T*>(bytes.data());
    }

    template <typename T>
    const T* data() const {
        return reinterpret_cast<const T*>(bytes.data());
    }

    size_t num_bytes() const { return bytes.size(); }
};

inline std::string trim_copy(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

inline std::vector<size_t> parse_shape(const std::string& header) {
    const auto open = header.find('(');
    const auto close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("Invalid .npy header: missing shape");
    }
    std::vector<size_t> shape;
    std::stringstream ss(header.substr(open + 1, close - open - 1));
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim_copy(item);
        if (!item.empty()) {
            shape.push_back(static_cast<size_t>(std::stoll(item)));
        }
    }
    return shape;
}

inline size_t product(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (size_t dim : shape) {
        n *= dim;
    }
    return n;
}

inline std::string header_descr_for(const std::type_info& ti, size_t bytes_per_item) {
    if (ti == typeid(double)) return "<f8";
    if (ti == typeid(float)) return "<f4";
    if (ti == typeid(std::int32_t)) return "<i4";
    if (ti == typeid(std::int64_t)) return "<i8";
    if (ti == typeid(std::uint32_t)) return "<u4";
    if (ti == typeid(std::uint64_t)) return "<u8";
    if (ti == typeid(std::int16_t)) return "<i2";
    if (ti == typeid(std::uint16_t)) return "<u2";
    if (ti == typeid(std::int8_t)) return "|i1";
    if (ti == typeid(std::uint8_t)) return "|u1";
    throw std::runtime_error("Unsupported numpy dtype size: " + std::to_string(bytes_per_item));
}

inline std::string header_descr_from_string(const std::string& descr) {
    return descr;
}

inline std::string build_header(const std::string& descr, const std::vector<size_t>& shape) {
    std::ostringstream oss;
    oss << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (shape.size() == 1 || i + 1 < shape.size()) {
            oss << ", ";
        }
    }
    oss << "), }";
    std::string header = oss.str();
    while ((10 + header.size()) % 16 != 0) {
        header.push_back(' ');
    }
    header.push_back('\n');
    return header;
}

inline NpyArray npy_load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open numpy file: " + filename);
    }

    char magic[6];
    in.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("Invalid numpy magic header: " + filename);
    }

    unsigned char major = 0, minor = 0;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16 = 0;
        in.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else if (major == 2) {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("Unsupported numpy format version in " + filename);
    }

    std::string header(header_len, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_len));
    auto shape = parse_shape(header);

    std::string descr;
    const auto descr_pos = header.find("'descr':");
    if (descr_pos == std::string::npos) {
        throw std::runtime_error("Missing dtype descriptor in " + filename);
    }
    const auto first_quote = header.find('\'', descr_pos + 8);
    const auto second_quote = header.find('\'', first_quote + 1);
    descr = header.substr(first_quote + 1, second_quote - first_quote - 1);

    if (header.find("fortran_order': True") != std::string::npos) {
        throw std::runtime_error("Fortran-ordered arrays are not supported: " + filename);
    }

    const size_t count = product(shape);
    const size_t item_size = descr.size() >= 2 ? static_cast<size_t>(std::stoul(descr.substr(2))) : 0;
    if (item_size == 0) {
        throw std::runtime_error("Could not parse numpy item size in " + filename);
    }

    NpyArray arr;
    arr.shape = std::move(shape);
    arr.bytes.resize(count * item_size);
    in.read(arr.bytes.data(), static_cast<std::streamsize>(arr.bytes.size()));
    if (static_cast<size_t>(in.gcount()) != arr.bytes.size()) {
        throw std::runtime_error("Unexpected EOF while reading numpy data: " + filename);
    }
    return arr;
}

template <typename T>
inline void npy_save(const std::string& filename, const T* data, const std::vector<size_t>& shape) {
    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open output numpy file: " + filename);
    }

    const std::string descr = header_descr_for(typeid(T), sizeof(T));
    const std::string header = build_header(descr, shape);

    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);
    const unsigned char major = 1;
    const unsigned char minor = 0;
    out.write(reinterpret_cast<const char*>(&major), 1);
    out.write(reinterpret_cast<const char*>(&minor), 1);
    const uint16_t header_len = static_cast<uint16_t>(header.size());
    out.write(reinterpret_cast<const char*>(&header_len), 2);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(product(shape) * sizeof(T)));
}

} // namespace cnpy
