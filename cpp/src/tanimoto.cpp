#include "rigid/tanimoto.h"

#include <limits>

namespace {
constexpr unsigned char nibble_count[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

constexpr unsigned int popcount(unsigned int value) {
    return nibble_count[value & 0x0FU] + nibble_count[value >> 4U];
}

bool product_overflows(size_t a, size_t b) {
    return b != 0 && a > std::numeric_limits<size_t>::max() / b;
}
}  // namespace

extern "C" RIGID_API int rigid_tanimoto_pairwise(
    const uint8_t* a, size_t rows_a, const uint8_t* b, size_t rows_b,
    size_t width, double* output) {
    if (product_overflows(rows_a, width) || product_overflows(rows_b, width)
        || product_overflows(rows_a, rows_b)
        || product_overflows(rows_a * rows_b, sizeof(double))
        || width > std::numeric_limits<size_t>::max() / 8U) {
        return 2;
    }
    if (rows_a == 0 || rows_b == 0) {
        return 0;
    }
    if (output == nullptr || (width != 0 && (a == nullptr || b == nullptr))) {
        return 1;
    }
    for (size_t i = 0; i < rows_a; ++i) {
        for (size_t j = 0; j < rows_b; ++j) {
            size_t intersection = 0;
            size_t union_count = 0;
            for (size_t k = 0; k < width; ++k) {
                const auto x = a[i * width + k];
                const auto y = b[j * width + k];
                intersection += popcount(x & y);
                union_count += popcount(x | y);
            }
            output[i * rows_b + j] = union_count == 0 ? 1.0
                : static_cast<double>(intersection) / static_cast<double>(union_count);
        }
    }
    return 0;
}
