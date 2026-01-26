#pragma once

#include <hip/hip_runtime.h>

#define XCD_NUM 8
#define CC_NUM  4 

#define L2_SIZE (4 * 1024 * 1024) // 4 MB
#define LLC_SIZE (256 * 1024 * 1024) // 256 MB

// GPU error check
#ifndef gpuErrchk
#define gpuErrchk(ans) { hipAssert((ans), __FILE__, __LINE__); }
inline void hipAssert(hipError_t code, const char *file, int line, bool abort=true){
    if (code != hipSuccess) {
        fprintf(stderr,"HIPassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif

inline uint32_t get_cc(uint32_t xcc_id) {
    return (xcc_id / (XCD_NUM / CC_NUM)) % CC_NUM;
}