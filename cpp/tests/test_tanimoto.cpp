#include "rigid/tanimoto.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
}

int main() {
    try {
        const uint8_t fingerprints[] = {0, 0, 255, 0, 15, 0, 0, 255};
        double matrix[16] = {};
        require(rigid_tanimoto_pairwise(fingerprints, 4, fingerprints, 4, 2, matrix) == 0, "kernel status");
        for (size_t i = 0; i < 4; ++i) {
            require(matrix[i * 4 + i] == 1.0, "diagonal");
            for (size_t j = 0; j < 4; ++j) {
                require(matrix[i * 4 + j] == matrix[j * 4 + i], "symmetry");
            }
        }
        require(std::abs(matrix[6] - 0.5) < 1e-15, "half overlap");
        require(matrix[7] == 0.0, "disjoint bits");
        require(rigid_tanimoto_pairwise(nullptr, 0, nullptr, 3, 2, nullptr) == 0, "zero rows");
        require(rigid_tanimoto_pairwise(nullptr, 1, fingerprints, 1, 2, matrix) == 1, "null input");
        require(rigid_tanimoto_pairwise(fingerprints, 1, fingerprints, 1, 2, nullptr) == 1, "null output");
        require(rigid_tanimoto_pairwise(nullptr, 1, nullptr, 1, 0, matrix) == 0 && matrix[0] == 1.0, "empty bits");
        require(rigid_tanimoto_pairwise(nullptr, std::numeric_limits<size_t>::max(), nullptr, 2, 2, matrix) == 2, "overflow guard");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
