#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cassert>

void calculateHistogramCPU(const std::vector<float>& imageData, std::vector<int>& histogram, 
                           size_t totalPixels, int channels, int numBins) {
    std::fill(histogram.begin(), histogram.end(), 0);
    #pragma omp parallel
    {
        std::vector<int> local_hist(numBins, 0);
        #pragma omp for nowait
        for (size_t i = 0; i < totalPixels; ++i) {
            float brightness = 0.0f;
            size_t pixelOffset = i * channels;
            for (int c = 0; c < channels; ++c) brightness += imageData[pixelOffset + c];
            brightness /= channels;
            int bin = static_cast<int>(brightness * (numBins - 1));
            local_hist[bin]++;
        }
        #pragma omp critical
        {
            for (int b = 0; b < numBins; ++b) histogram[b] += local_hist[b];
        }
    }
}