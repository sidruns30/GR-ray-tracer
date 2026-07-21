#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <vector>

namespace cnpy {

struct NpyArray {
    std::vector<size_t> shape;
    std::string descr;
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

struct NpyHeader {
    std::vector<size_t> shape;
    std::string descr;
    size_t item_size = 0;
    std::streamoff data_offset = 0;
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

template <typename Integer>
inline void append_little_endian(std::vector<char>& bytes, Integer value) {
    static_assert(std::is_integral_v<Integer>);
    using Unsigned = std::make_unsigned_t<Integer>;
    const Unsigned unsigned_value = static_cast<Unsigned>(value);
    for (std::size_t i = 0; i < sizeof(Integer); ++i) {
        bytes.push_back(static_cast<char>((unsigned_value >> (8 * i)) & 0xffu));
    }
}

template <typename T>
inline std::vector<char> make_npy_bytes(const T* data, const std::vector<size_t>& shape) {
    const std::string header = build_header(header_descr_for(typeid(T), sizeof(T)), shape);
    if (header.size() > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("NumPy header is too large for format version 1");
    }

    std::vector<char> bytes;
    const std::size_t data_size = product(shape) * sizeof(T);
    bytes.reserve(10 + header.size() + data_size);
    const char magic[] = "\x93NUMPY";
    bytes.insert(bytes.end(), magic, magic + 6);
    bytes.push_back(1);
    bytes.push_back(0);
    append_little_endian(bytes, static_cast<std::uint16_t>(header.size()));
    bytes.insert(bytes.end(), header.begin(), header.end());
    if (data_size > 0) {
        if (!data) throw std::runtime_error("Cannot serialize a null NumPy data pointer");
        const char* raw_data = reinterpret_cast<const char*>(data);
        bytes.insert(bytes.end(), raw_data, raw_data + data_size);
    }
    return bytes;
}

inline std::uint32_t crc32(const std::vector<char>& bytes) {
    std::uint32_t crc = 0xffffffffu;
    for (const unsigned char byte : bytes) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

// Writes an uncompressed .npz archive. NumPy reads stored ZIP members directly,
// and avoiding compression keeps checkpoints fast and dependency-free.
class NpzWriter {
public:
    explicit NpzWriter(const std::string& filename)
        : out_(filename, std::ios::binary | std::ios::trunc) {
        if (!out_) {
            throw std::runtime_error("Failed to open output NumPy archive: " + filename);
        }
    }

    NpzWriter(const NpzWriter&) = delete;
    NpzWriter& operator=(const NpzWriter&) = delete;

    template <typename T>
    void add(const std::string& name, const T* data, const std::vector<size_t>& shape) {
        if (closed_) {
            throw std::runtime_error("Cannot add an array to a closed NumPy archive");
        }

        const std::string member_name = name + ".npy";
        const std::vector<char> payload = make_npy_bytes(data, shape);
        if (member_name.size() > std::numeric_limits<std::uint16_t>::max() ||
            payload.size() > std::numeric_limits<std::uint32_t>::max() ||
            offset_ > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("NumPy archive exceeds the non-ZIP64 size limit");
        }

        Entry entry{
            member_name,
            crc32(payload),
            static_cast<std::uint32_t>(payload.size()),
            static_cast<std::uint32_t>(offset_)
        };

        std::vector<char> header;
        append_little_endian(header, std::uint32_t{0x04034b50});
        append_little_endian(header, std::uint16_t{20});
        append_little_endian(header, std::uint16_t{0});
        append_little_endian(header, std::uint16_t{0});
        append_little_endian(header, std::uint16_t{0});
        append_little_endian(header, std::uint16_t{0});
        append_little_endian(header, entry.crc);
        append_little_endian(header, entry.size);
        append_little_endian(header, entry.size);
        append_little_endian(header, static_cast<std::uint16_t>(member_name.size()));
        append_little_endian(header, std::uint16_t{0});

        write(header);
        out_.write(member_name.data(), static_cast<std::streamsize>(member_name.size()));
        out_.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        ensure_write_succeeded();
        offset_ += header.size() + member_name.size() + payload.size();
        entries_.push_back(std::move(entry));
    }

    void close() {
        if (closed_) return;
        if (entries_.size() > std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("NumPy archive contains too many arrays");
        }

        const std::uint64_t central_directory_offset = offset_;
        for (const Entry& entry : entries_) {
            std::vector<char> header;
            append_little_endian(header, std::uint32_t{0x02014b50});
            append_little_endian(header, std::uint16_t{20});
            append_little_endian(header, std::uint16_t{20});
            append_little_endian(header, std::uint16_t{0});
            append_little_endian(header, std::uint16_t{0});
            append_little_endian(header, std::uint16_t{0});
            append_little_endian(header, std::uint16_t{0});
            append_little_endian(header, entry.crc);
            append_little_endian(header, entry.size);
            append_little_endian(header, entry.size);
            append_little_endian(header, static_cast<std::uint16_t>(entry.name.size()));
            append_little_endian(header, std::uint16_t{0});
            append_little_endian(header, std::uint16_t{0});
            append_little_endian(header, std::uint16_t{0});
            append_little_endian(header, std::uint16_t{0});
            append_little_endian(header, std::uint32_t{0});
            append_little_endian(header, entry.local_header_offset);
            write(header);
            out_.write(entry.name.data(), static_cast<std::streamsize>(entry.name.size()));
            ensure_write_succeeded();
            offset_ += header.size() + entry.name.size();
        }

        const std::uint64_t central_directory_size = offset_ - central_directory_offset;
        if (central_directory_offset > std::numeric_limits<std::uint32_t>::max() ||
            central_directory_size > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("NumPy archive exceeds the non-ZIP64 size limit");
        }

        std::vector<char> end;
        append_little_endian(end, std::uint32_t{0x06054b50});
        append_little_endian(end, std::uint16_t{0});
        append_little_endian(end, std::uint16_t{0});
        append_little_endian(end, static_cast<std::uint16_t>(entries_.size()));
        append_little_endian(end, static_cast<std::uint16_t>(entries_.size()));
        append_little_endian(end, static_cast<std::uint32_t>(central_directory_size));
        append_little_endian(end, static_cast<std::uint32_t>(central_directory_offset));
        append_little_endian(end, std::uint16_t{0});
        write(end);
        out_.close();
        if (!out_) {
            throw std::runtime_error("Failed to finalize NumPy archive");
        }
        closed_ = true;
    }

    ~NpzWriter() {
        if (!closed_) {
            try {
                close();
            } catch (...) {
            }
        }
    }

private:
    struct Entry {
        std::string name;
        std::uint32_t crc;
        std::uint32_t size;
        std::uint32_t local_header_offset;
    };

    void write(const std::vector<char>& bytes) {
        out_.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        ensure_write_succeeded();
    }

    void ensure_write_succeeded() const {
        if (!out_) {
            throw std::runtime_error("Failed while writing NumPy archive");
        }
    }

    std::ofstream out_;
    std::vector<Entry> entries_;
    std::uint64_t offset_ = 0;
    bool closed_ = false;
};

inline NpyHeader read_npy_header(std::istream& in, const std::string& filename) {
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
    NpyHeader result;
    result.shape = parse_shape(header);

    const auto descr_pos = header.find("'descr':");
    if (descr_pos == std::string::npos) {
        throw std::runtime_error("Missing dtype descriptor in " + filename);
    }
    const auto first_quote = header.find('\'', descr_pos + 8);
    const auto second_quote = header.find('\'', first_quote + 1);
    result.descr = header.substr(first_quote + 1, second_quote - first_quote - 1);

    if (header.find("fortran_order': True") != std::string::npos) {
        throw std::runtime_error("Fortran-ordered arrays are not supported: " + filename);
    }

    result.item_size = result.descr.size() >= 2
        ? static_cast<size_t>(std::stoul(result.descr.substr(2))) : 0;
    if (result.item_size == 0) {
        throw std::runtime_error("Could not parse numpy item size in " + filename);
    }
    result.data_offset = in.tellg();
    if (!in) throw std::runtime_error("Failed to read numpy header: " + filename);
    return result;
}

inline NpyArray npy_load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open numpy file: " + filename);
    }
    NpyHeader header = read_npy_header(in, filename);

    NpyArray arr;
    arr.shape = std::move(header.shape);
    arr.descr = std::move(header.descr);
    arr.bytes.resize(product(arr.shape) * header.item_size);
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

    const std::vector<char> bytes = make_npy_bytes(data, shape);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        throw std::runtime_error("Failed while writing numpy file: " + filename);
    }
}

} // namespace cnpy
