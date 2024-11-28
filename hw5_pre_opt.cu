/*
    Name: Chase Compton
    Email: cscompton1@crimson.ua.edu
    Course Section: CS 481
    Homework #5: Game of Life using GPU (CUDA)

    Implements a GPU version of the "Game of Life" program

    Instructions to compile the program:

    Instructions to run the program:
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <string.h>

#define DIES 0
#define ALIVE 1

// Function to get current time in seconds
double gettime(void)
{
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return ((double)tval.tv_sec + (double)tval.tv_usec / 1000000.0);
}

// Function to print the current state of the Game of Life board
void printarray(int **a, int nRows, int nCols, int k, FILE *outfile)
{
    int i, j;
    char *border = (char *)malloc(sizeof(char) * (nCols * 2 + 4));
    memset(border, '-', nCols * 2 + 3);
    border[nCols * 2 + 3] = '\0';

    if (outfile == stdout)
    {
        printf("Life after %d iterations:\n", k);
    }
    else
    {
        fprintf(outfile, "Final state after %d iterations:\n", k);
    }

    fprintf(outfile, "%s\n", border);
    for (i = 1; i <= nRows; i++)
    {
        fprintf(outfile, "| ");
        for (j = 1; j <= nCols; j++)
            fprintf(outfile, "%s ", a[i][j] ? "■" : "□");
        fprintf(outfile, "|\n");
    }
    fprintf(outfile, "%s\n\n", border);

    free(border);
}

// Kernel function to compute next generation and detect changes
__global__ void compute_kernel(int *d_life, int *d_temp, int N, int *d_flag)
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
        if (ty == 0 && i > 1)
            s_life[0 * s_width + (tx + 1)] = d_life[(i - 1) * (N + 2) + j];
        if (ty == blockDim.y - 1 && i < N)
            s_life[(ty + 2) * s_width + (tx + 1)] = d_life[(i + 1) * (N + 2) + j];
        if (tx == 0 && j > 1)
            s_life[(ty + 1) * s_width + 0] = d_life[i * (N + 2) + (j - 1)];
        if (tx == blockDim.x - 1 && j < N)
            s_life[(ty + 1) * s_width + (tx + 2)] = d_life[i * (N + 2) + (j + 1)];

        // Load corner cells
        if (tx == 0 && ty == 0 && i > 1 && j > 1)
            s_life[0 * s_width + 0] = d_life[(i - 1) * (N + 2) + (j - 1)];
        if (tx == blockDim.x - 1 && ty == 0 && i > 1 && j < N)
            s_life[0 * s_width + (tx + 2)] = d_life[(i - 1) * (N + 2) + (j + 1)];
        if (tx == 0 && ty == blockDim.y - 1 && i < N && j > 1)
            s_life[(ty + 2) * s_width + 0] = d_life[(i + 1) * (N + 2) + (j - 1)];
        if (tx == blockDim.x - 1 && ty == blockDim.y - 1 && i < N && j < N)
            s_life[(ty + 2) * s_width + (tx + 2)] = d_life[(i + 1) * (N + 2) + (j + 1)];
    }

    __syncthreads();

    if (i <= N && j <= N)
    {
        // Calculate the sum of neighbors
        int sum = 0;
        sum += s_life[(ty)*s_width + (tx)];
        sum += s_life[(ty)*s_width + (tx + 1)];
        sum += s_life[(ty)*s_width + (tx + 2)];
        sum += s_life[(ty + 1) * s_width + (tx)];
        sum += s_life[(ty + 1) * s_width + (tx + 2)];
        sum += s_life[(ty + 2) * s_width + (tx)];
        sum += s_life[(ty + 2) * s_width + (tx + 1)];
        sum += s_life[(ty + 2) * s_width + (tx + 2)];

        // Apply the Game of Life rules
        int new_state = s_life[(ty + 1) * s_width + (tx + 1)];
        if (new_state == ALIVE)
        {
            if (sum < 2 || sum > 3)
            {
                new_state = DIES;
            }
        }
        else
        {
            if (sum == 3)
            {
                new_state = ALIVE;
            }
        }

        // If the state changes, set flag
        if (new_state != s_life[(ty + 1) * s_width + (tx + 1)])
        {
            atomicAdd(d_flag, 1);
        }

        // Write new state to d_temp
        d_temp[idx] = new_state;
    }
}

int main(int argc, char **argv)
{
    int N, NTIMES;
    int i, j, k;
    int **life = NULL;
    double t1, t2;
    char *output_dir;
    FILE *outfile = NULL;

    if (argc != 4)
    {
        printf("Usage: %s <size> <max_iterations> <output_directory>\n", argv[0]);
        exit(-1);
    }

    N = atoi(argv[1]);
    NTIMES = atoi(argv[2]);
    output_dir = argv[3];

    // Open the output file
    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "%s/cuda_%d_%d_output.txt", output_dir, N, NTIMES);
    outfile = fopen(output_filename, "w");
    if (outfile == NULL)
    {
        printf("Error opening output file: %s\n", output_filename);
        exit(-1);
    }

    // Allocate host memory
    life = (int **)malloc((N + 2) * sizeof(int *));
    if (life == NULL)
    {
        fprintf(stderr, "Error allocating memory for life\n");
        exit(EXIT_FAILURE);
    }
    for (i = 0; i < N + 2; i++)
    {
        life[i] = (int *)malloc((N + 2) * sizeof(int));
        if (life[i] == NULL)
        {
            fprintf(stderr, "Error allocating memory for life[%d]\n", i);
            exit(EXIT_FAILURE);
        }
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
    if (h_life == NULL)
    {
        fprintf(stderr, "Error allocating memory for h_life\n");
        exit(EXIT_FAILURE);
    }

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
    cudaError_t err;
    err = cudaMalloc((void **)&d_life, (N + 2) * (N + 2) * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: Failed to allocate device memory for d_life: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_temp, (N + 2) * (N + 2) * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: Failed to allocate device memory for d_temp: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate device memory for flag
    int *d_flag;
    err = cudaMalloc((void **)&d_flag, sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: Failed to allocate device memory for d_flag: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy data to device
    err = cudaMemcpy(d_life, h_life, (N + 2) * (N + 2) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: Failed to copy data to d_life: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Define block and grid sizes
    int THREADS = 16; // Adjust as needed
    dim3 threadsPerBlock(THREADS, THREADS);
    dim3 numBlocks((N + THREADS - 1) / THREADS, (N + THREADS - 1) / THREADS);

    size_t shared_mem_size = (THREADS + 2) * (THREADS + 2) * sizeof(int);

    // Start the timer
    t1 = gettime();

    int h_flag = 1; // Host flag to check if any changes occurred
    for (k = 0; k < NTIMES && h_flag != 0; k++)
    {
        // Initialize device flag to zero
        err = cudaMemset(d_flag, 0, sizeof(int));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Error: Failed to set d_flag to zero: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Launch the kernel
        compute_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(d_life, d_temp, N, d_flag);

        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Error: Kernel launch failed: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Synchronize
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Error: cudaDeviceSynchronize returned error code %d after launching kernel!\n", err);
            exit(EXIT_FAILURE);
        }

        // Copy flag back to host
        err = cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Error: Failed to copy d_flag to host: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Swap d_life and d_temp
        int *temp_ptr = d_life;
        d_life = d_temp;
        d_temp = temp_ptr;
    }

    // Stop the timer
    t2 = gettime();
    printf("Time taken %f seconds for %d iterations on a board size of %d x %d\n", t2 - t1, k, N, N);
    fprintf(outfile, "Time taken %f seconds for %d iterations\n", t2 - t1, k);

    // Copy the result back to host
    err = cudaMemcpy(h_life, d_life, (N + 2) * (N + 2) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: Failed to copy data from d_life to host: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy data from h_life[] back to life[][]
    for (i = 0; i < N + 2; i++)
    {
        for (j = 0; j < N + 2; j++)
        {
            life[i][j] = h_life[i * (N + 2) + j];
        }
    }

    // Print final state to output file
    printarray(life, N, N, k, outfile);

    // Close the output file
    fclose(outfile);

    // Free memory
    for (i = 0; i < N + 2; i++)
    {
        free(life[i]);
    }
    free(life);
    free(h_life);
    cudaFree(d_life);
    cudaFree(d_temp);
    cudaFree(d_flag);

    return 0;
}