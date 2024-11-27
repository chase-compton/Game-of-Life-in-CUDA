/*
    Name: Chase Compton
    Email: cscompton1@crimson.ua.edu
    Course Section: CS 481
    Homework #5: Game of Life using GPU (CUDA)

    Implements a GPU version of the "Game of Life" program in C

    Instructions to compile the program:

    Instructions to run the program:
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>

#define DIES 0
#define ALIVE 1

// Function to get current time in seconds
double gettime(void)
{
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return ((double)tval.tv_sec + (double)tval.tv_usec / 1000000.0);
}

// Kernel function to compute next generation
__global__ void compute_kernel(int *d_life, int *d_temp, int N)
{
    extern __shared__ int s_life[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = blockIdx.y * blockDim.y + ty + 1;
    int j = blockIdx.x * blockDim.x + tx + 1;

    int s_width = blockDim.x + 2; // Shared memory width with halo

    // Global index
    int idx = i * (N + 2) + j;

    // Load data into shared memory with halo cells
    if (i <= N && j <= N)
    {
        // Load center cell
        s_life[(ty + 1) * s_width + (tx + 1)] = d_life[idx];

        // Load halo cells
        if (ty == 0)
            s_life[0 * s_width + (tx + 1)] = d_life[(i - 1) * (N + 2) + j];
        if (ty == blockDim.y - 1)
            s_life[(ty + 2) * s_width + (tx + 1)] = d_life[(i + 1) * (N + 2) + j];
        if (tx == 0)
            s_life[(ty + 1) * s_width + 0] = d_life[i * (N + 2) + (j - 1)];
        if (tx == blockDim.x - 1)
            s_life[(ty + 1) * s_width + (tx + 2)] = d_life[i * (N + 2) + (j + 1)];

        // Load corner cells
        if (tx == 0 && ty == 0)
            s_life[0 * s_width + 0] = d_life[(i - 1) * (N + 2) + (j - 1)];
        if (tx == blockDim.x - 1 && ty == 0)
            s_life[0 * s_width + (tx + 2)] = d_life[(i - 1) * (N + 2) + (j + 1)];
        if (tx == 0 && ty == blockDim.y - 1)
            s_life[(ty + 2) * s_width + 0] = d_life[(i + 1) * (N + 2) + (j - 1)];
        if (tx == blockDim.x - 1 && ty == blockDim.y - 1)
            s_life[(ty + 2) * s_width + (tx + 2)] = d_life[(i + 1) * (N + 2) + (j + 1)];
    }

    __syncthreads();

    if (i <= N && j <= N)
    {
        // Calculate the sum of neighbors
        int sum = s_life[(ty) * s_width + (tx)] + s_life[(ty) * s_width + (tx + 1)] + s_life[(ty) * s_width + (tx + 2)] +
                  s_life[(ty + 1) * s_width + (tx)] + s_life[(ty + 1) * s_width + (tx + 2)] +
                  s_life[(ty + 2) * s_width + (tx)] + s_life[(ty + 2) * s_width + (tx + 1)] + s_life[(ty + 2) * s_width + (tx + 2)];

        // Apply the Game of Life rules
        if (s_life[(ty + 1) * s_width + (tx + 1)] == ALIVE)
        {
            if (sum < 2 || sum > 3)
                d_temp[idx] = DIES;
            else
                d_temp[idx] = ALIVE;
        }
        else
        {
            if (sum == 3)
                d_temp[idx] = ALIVE;
            else
                d_temp[idx] = DIES;
        }
    }
}

int main(int argc, char **argv)
{
    int N, NTIMES;
    int i, j, k, flag = 1;
    int **life = NULL;
    double t1, t2;

    if (argc != 3)
    {
        printf("Usage: %s <size> <max_iterations>\n", argv[0]);
        exit(-1);
    }

    N = atoi(argv[1]);
    NTIMES = atoi(argv[2]);

    // Allocate host memory
    life = (int **)malloc((N + 2) * sizeof(int *));
    for (i = 0; i < N + 2; i++)
    {
        life[i] = (int *)malloc((N + 2) * sizeof(int));
    }

    // Initialize the game board with random live/dead cells
    srand(54321);
    for (i = 1; i <= N; i++)
    {
        for (j = 1; j <= N; j++)
        {
            if (drand48() < 0.5)
                life[i][j] = ALIVE;
            else
                life[i][j] = DIES;
        }
    }

    // Initialize the boundaries to DIES
    for (i = 0; i < N + 2; i++)
    {
        life[0][i] = life[N + 1][i] = DIES;
        life[i][0] = life[i][N + 1] = DIES;
    }

    // Flatten the 2D arrays into 1D
    int *h_life = (int *)malloc((N + 2) * (N + 2) * sizeof(int));

    // Copy data from life[][] to h_life[]
    for (i = 0; i < N + 2; i++)
    {
        for (j = 0; j < N + 2; j++)
        {
            h_life[i * (N + 2) + j] = life[i][j];
        }
    }

    // Allocate device memory
    int *d_life, *d_temp;
    cudaMalloc((void **)&d_life, (N + 2) * (N + 2) * sizeof(int));
    cudaMalloc((void **)&d_temp, (N + 2) * (N + 2) * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_life, h_life, (N + 2) * (N + 2) * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int THREADS = 16; // Adjust as needed
    dim3 threadsPerBlock(THREADS, THREADS);
    dim3 numBlocks((N + THREADS - 1) / THREADS, (N + THREADS - 1) / THREADS);

    size_t shared_mem_size = (THREADS + 2) * (THREADS + 2) * sizeof(int);

    // Start the timer
    t1 = gettime();

    for (k = 0; k < NTIMES && flag != 0; k++)
    {
        // Launch the kernel
        compute_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(d_life, d_temp, N);

        // Synchronize
        cudaDeviceSynchronize();

        // Swap d_life and d_temp
        int *temp_ptr = d_life;
        d_life = d_temp;
        d_temp = temp_ptr;

        // For now, we assume flag != 0
    }

    // Stop the timer
    t2 = gettime();
    printf("Time taken %f seconds for %d iterations on a board size of %d x %d\n", t2 - t1, k, N, N);

    // Copy the result back to host
    cudaMemcpy(h_life, d_life, (N + 2) * (N + 2) * sizeof(int), cudaMemcpyDeviceToHost);

    // Copy data from h_life[] back to life[][]
    for (i = 0; i < N + 2; i++)
    {
        for (j = 0; j < N + 2; j++)
        {
            life[i][j] = h_life[i * (N + 2) + j];
        }
    }

    // Free memory
    for (i = 0; i < N + 2; i++)
    {
        free(life[i]);
    }
    free(life);
    free(h_life);
    cudaFree(d_life);
    cudaFree(d_temp);

    return 0;
}