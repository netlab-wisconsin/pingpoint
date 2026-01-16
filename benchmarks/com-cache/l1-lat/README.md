### troubleshooting

- Using ARRAY_SIZE directly inside the kernel, instead of passing it as a parameter, somehow avoids L1 cache access while initializing the pointer-chasing array and results in reduced TCP_TOTAL_CACHE_ACCESSES_sum in rocprof output. While this does not affect the latency measurement, try to pass by parameter when verifying the memory access pattern using rocprof.