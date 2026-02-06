#!/bin/bash

while true; do 
    # Optional: Print timestamp so you know WHEN the spike happened
    # date >> results/memory_ov.out
    
    # Run command and append to file
    rocm-smi --showmeminfo vram >> results/memory_ov.out
    
    # Wait
    sleep 0.001
done