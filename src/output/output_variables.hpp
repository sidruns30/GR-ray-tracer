#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

inline constexpr std::array<std::string_view, 20> available_output_variables = {
    "id", "frequency", "emission_frame_energy",
    "x0", "x1", "x2", "x3",
    "k0", "k1", "k2", "k3",
    "I", "Q", "U", "V",
    "dlambda", "terminate", "phase", "theta_disp", "phi_disp"
};

inline constexpr std::string_view default_output_variables =
    "id,frequency,"
    "x0,x1,x2,x3,k0,k1,k2,k3,I,Q,U,V,phase";

class OutputSelection {
public:
    static OutputSelection parse(std::string specification) {
        trim(specification);
        if (specification.empty()) {
            throw std::runtime_error("output variables cannot be empty");
        }

        if (specification.front() == '[' && specification.back() == ']') {
            specification = specification.substr(1, specification.size() - 2);
        }

        OutputSelection selection;
        std::size_t begin = 0;
        while (begin <= specification.size()) {
            const std::size_t comma = specification.find(',', begin);
            std::string name = specification.substr(begin, comma - begin);
            trim(name);
            if (name.size() >= 2 &&
                ((name.front() == '"' && name.back() == '"') ||
                 (name.front() == '\'' && name.back() == '\''))) {
                name = name.substr(1, name.size() - 2);
            }
            trim(name);

            if (name == "all") {
                for (const std::string_view variable : available_output_variables) {
                    selection.names_.emplace_back(variable);
                }
                return selection;
            }
            if (!name.empty()) {
                if (!is_available(name)) {
                    throw std::runtime_error("unknown output variable: " + name);
                }
                if (!selection.contains(name)) {
                    selection.names_.push_back(std::move(name));
                }
            }
            if (comma == std::string::npos) break;
            begin = comma + 1;
        }

        if (selection.names_.empty()) {
            throw std::runtime_error("output variables cannot be empty");
        }
        return selection;
    }

    bool contains(std::string_view name) const {
        return std::find(names_.begin(), names_.end(), name) != names_.end();
    }

    const std::vector<std::string>& names() const { return names_; }

private:
    static void trim(std::string& value) {
        const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
            return std::isspace(c);
        });
        const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
            return std::isspace(c);
        }).base();
        value = first < last ? std::string(first, last) : std::string{};
    }

    static bool is_available(std::string_view name) {
        return std::find(available_output_variables.begin(), available_output_variables.end(), name) !=
               available_output_variables.end();
    }

    std::vector<std::string> names_;
};
