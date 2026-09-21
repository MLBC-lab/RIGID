#ifndef RIGID_TANIMOTO_H
#define RIGID_TANIMOTO_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(RIGID_BUILD_SHARED)
#define RIGID_API __declspec(dllexport)
#elif defined(_WIN32) && defined(RIGID_USE_SHARED)
#define RIGID_API __declspec(dllimport)
#else
#define RIGID_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Row-major uint8 fingerprints and float64 output. Status: 0 success,
 * 1 invalid pointer, 2 dimensions overflow. Empty/empty similarity is 1. */
RIGID_API int rigid_tanimoto_pairwise(
    const uint8_t* a, size_t rows_a,
    const uint8_t* b, size_t rows_b,
    size_t bytes_per_fingerprint, double* output);

#ifdef __cplusplus
}
#endif

#endif
