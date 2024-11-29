# Game of Life using GPU (CUDA)

## Author
- **Name:** Chase Compton  
- **Email:** cscompton1@crimson.ua.edu  

## Course Information
- **Course Section:** CS 481  
- **Assignment:** Homework #5  

## Project Description
This project implements a GPU-accelerated version of Conway's **Game of Life**, featuring:  
- **Bit-packing** to optimize memory usage.  
- **Shared memory** for efficient GPU memory management.  
- **Loop unrolling** to improve computation speed.  

## Prerequisites
- **CUDA Toolkit** installed on your system.  
- Compatible **NVIDIA GPU**.  
- **Nsight Systems (nsys)** profiling tool (optional, for profiling).  

## Compilation Instructions
To compile the program, use the following command:  
```bash
nvcc -O3 -o hw5 hw5.cu
```

## How to Run
To run the program, use the following command:  
```bash
nsys profile --stats=true ./hw5 <size> <max_iterations> <output_directory>
```