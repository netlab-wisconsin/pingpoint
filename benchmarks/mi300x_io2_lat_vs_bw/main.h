#pragma once

#include <hip/hip_runtime.h>

#define XCD_NUM 8
#define CC_NUM  4 

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