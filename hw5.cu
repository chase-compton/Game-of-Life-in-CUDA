/*
    Name: Chase Compton
    Email: cscompton1@crimson.ua.edu
    Course Section: CS 481
    Homework #5: Game of Life using GPU (CUDA)

    Implements a GPU version of the "Game of Life" program with bit-packing, shared memory, and loop unrolling optimizations.

    Instructions to compile the program: nvcc -O3 -o hw5 hw5.cu

    Instructions to run the program: nsys profile --stats=true ./hw5 <size> <max_iterations> <output_directory>
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <string.h>

#define DIES 0
#define ALIVE 1

#define WORD unsigned int
#define WORD_SIZE (sizeof(WORD) * 8)

// Function to get current time in seconds
double gettime(void)
{
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return ((double)tval.tv_sec + (double)tval.tv_usec / 1000000.0);
}

// Function to print the current state of the Game of Life board
void printarray(WORD *a, int nRows, int nCols, int words_per_row, int k, FILE *outfile)
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
        {
            int idx = i * words_per_row + ((j - 1) / WORD_SIZE);
            int bit = (j - 1) % WORD_SIZE;
            int cell_state = (a[idx] >> bit) & 1;
            fprintf(outfile, "%s ", cell_state ? "■" : "□");
        }
        fprintf(outfile, "|\n");
    }
    fprintf(outfile, "%s\n\n", border);

    free(border);
}

// Kernel function to compute next generation and detect changes
__global__ void compute_kernel(WORD *d_life, WORD *d_temp, int N, int words_per_row, int *d_flag)
{
    extern __shared__ WORD s_life[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int THREADS_X = blockDim.x;
    int THREADS_Y = blockDim.y;

    // Compute global indices
    int global_i = by * THREADS_Y + ty + 1; // +1 for halo
    int global_j = bx * THREADS_X + tx;     // word index

    // Compute shared memory indices
    int shared_i = ty + 1; // +1 for halo
    int shared_j = tx + 1; // +1 for halo

    int shared_words_per_row = THREADS_X + 2; // +2 for halos

    // Load data into shared memory, including halos
    // Load center data
    if (global_i <= N && global_j < words_per_row)
    {
        s_life[shared_i * shared_words_per_row + shared_j] = d_life[global_i * words_per_row + global_j];
    }
    else
    {
        s_life[shared_i * shared_words_per_row + shared_j] = 0;
    }

    // Load halos in x-direction
    // Left halo
    if (tx == 0)
    {
        if (global_j > 0 && global_i <= N)
        {
            s_life[shared_i * shared_words_per_row + (shared_j - 1)] = d_life[global_i * words_per_row + (global_j - 1)];
        }
        else
        {
            s_life[shared_i * shared_words_per_row + (shared_j - 1)] = 0;
        }
    }
    // Right halo
    if (tx == THREADS_X - 1)
    {
        if (global_j + 1 < words_per_row && global_i <= N)
        {
            s_life[shared_i * shared_words_per_row + (shared_j + 1)] = d_life[global_i * words_per_row + (global_j + 1)];
        }
        else
        {
            s_life[shared_i * shared_words_per_row + (shared_j + 1)] = 0;
        }
    }

    // Load halos in y-direction
    // Top halo
    if (ty == 0)
    {
        if (global_i > 1 && global_j < words_per_row)
        {
            s_life[(shared_i - 1) * shared_words_per_row + shared_j] = d_life[(global_i - 1) * words_per_row + global_j];
        }
        else
        {
            s_life[(shared_i - 1) * shared_words_per_row + shared_j] = 0;
        }

        // Corners
        if (tx == 0)
        {
            if (global_j > 0 && global_i > 1)
            {
                s_life[(shared_i - 1) * shared_words_per_row + (shared_j - 1)] = d_life[(global_i - 1) * words_per_row + (global_j - 1)];
            }
            else
            {
                s_life[(shared_i - 1) * shared_words_per_row + (shared_j - 1)] = 0;
            }
        }
        if (tx == THREADS_X - 1)
        {
            if (global_j + 1 < words_per_row && global_i > 1)
            {
                s_life[(shared_i - 1) * shared_words_per_row + (shared_j + 1)] = d_life[(global_i - 1) * words_per_row + (global_j + 1)];
            }
            else
            {
                s_life[(shared_i - 1) * shared_words_per_row + (shared_j + 1)] = 0;
            }
        }
    }
    // Bottom halo
    if (ty == THREADS_Y - 1)
    {
        if (global_i + 1 <= N && global_j < words_per_row)
        {
            s_life[(shared_i + 1) * shared_words_per_row + shared_j] = d_life[(global_i + 1) * words_per_row + global_j];
        }
        else
        {
            s_life[(shared_i + 1) * shared_words_per_row + shared_j] = 0;
        }

        // Corners
        if (tx == 0)
        {
            if (global_j > 0 && global_i + 1 <= N)
            {
                s_life[(shared_i + 1) * shared_words_per_row + (shared_j - 1)] = d_life[(global_i + 1) * words_per_row + (global_j - 1)];
            }
            else
            {
                s_life[(shared_i + 1) * shared_words_per_row + (shared_j - 1)] = 0;
            }
        }
        if (tx == THREADS_X - 1)
        {
            if (global_j + 1 < words_per_row && global_i + 1 <= N)
            {
                s_life[(shared_i + 1) * shared_words_per_row + (shared_j + 1)] = d_life[(global_i + 1) * words_per_row + (global_j + 1)];
            }
            else
            {
                s_life[(shared_i + 1) * shared_words_per_row + (shared_j + 1)] = 0;
            }
        }
    }

    __syncthreads();

    // Now process the bits
    if (global_i >= 1 && global_i <= N && global_j < words_per_row)
    {
        int idx = global_i * words_per_row + global_j;

        WORD current_word = s_life[shared_i * shared_words_per_row + shared_j];
        WORD new_word = 0;

        // Neighbor words from shared memory
        WORD north_word = s_life[(shared_i - 1) * shared_words_per_row + shared_j];
        WORD south_word = s_life[(shared_i + 1) * shared_words_per_row + shared_j];
        WORD west_word = s_life[shared_i * shared_words_per_row + (shared_j - 1)];
        WORD east_word = s_life[shared_i * shared_words_per_row + (shared_j + 1)];

        WORD nw_word = s_life[(shared_i - 1) * shared_words_per_row + (shared_j - 1)];
        WORD ne_word = s_life[(shared_i - 1) * shared_words_per_row + (shared_j + 1)];
        WORD sw_word = s_life[(shared_i + 1) * shared_words_per_row + (shared_j - 1)];
        WORD se_word = s_life[(shared_i + 1) * shared_words_per_row + (shared_j + 1)];

        // Process each bit in the word
        for (int bit = 0; bit < WORD_SIZE; bit++)
        {
            int col = global_j * WORD_SIZE + bit + 1; // +1 for halo

            if (col >= 1 && col <= N)
            {
                int current_state = (current_word >> bit) & 1;
                int sum = 0;

                // Neighbor bits
                int w_bit = bit - 1;
                int e_bit = bit + 1;

                // West neighbor
                int west_state;
                if (w_bit >= 0)
                    west_state = (current_word >> w_bit) & 1;
                else
                    west_state = (west_word >> (WORD_SIZE - 1)) & 1;
                sum += west_state;

                // East neighbor
                int east_state;
                if (e_bit < WORD_SIZE)
                    east_state = (current_word >> e_bit) & 1;
                else
                    east_state = (east_word >> 0) & 1;
                sum += east_state;

                // North neighbor
                int north_state = (north_word >> bit) & 1;
                sum += north_state;

                // South neighbor
                int south_state = (south_word >> bit) & 1;
                sum += south_state;

                // North-West neighbor
                int nw_state;
                if (w_bit >= 0)
                    nw_state = (north_word >> w_bit) & 1;
                else
                    nw_state = (nw_word >> (WORD_SIZE - 1)) & 1;
                sum += nw_state;

                // North-East neighbor
                int ne_state;
                if (e_bit < WORD_SIZE)
                    ne_state = (north_word >> e_bit) & 1;
                else
                    ne_state = (ne_word >> 0) & 1;
                sum += ne_state;

                // South-West neighbor
                int sw_state;
                if (w_bit >= 0)
                    sw_state = (south_word >> w_bit) & 1;
                else
                    sw_state = (sw_word >> (WORD_SIZE - 1)) & 1;
                sum += sw_state;

                // South-East neighbor
                int se_state;
                if (e_bit < WORD_SIZE)
                    se_state = (south_word >> e_bit) & 1;
                else
                    se_state = (se_word >> 0) & 1;
                sum += se_state;

                // Apply the Game of Life rules
                int new_state = current_state;
                if (current_state == ALIVE)
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
                if (new_state != current_state)
                {
                    atomicAdd(d_flag, 1);
                }

                // Update the new word
                if (new_state == ALIVE)
                    new_word |= ((WORD)1 << bit);
                else
                    new_word &= ~((WORD)1 << bit);
            }
        }

        // Write the new word to global memory
        d_temp[idx] = new_word;
    }
}

int main(int argc, char **argv)
{
    int N, NTIMES;
    int i, j, k;
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

    // Adjust the board dimensions
    int total_rows = N + 2; // Including halos
    int total_cols = N + 2; // Including halos
    int padded_cols = ((total_cols + WORD_SIZE - 1) / WORD_SIZE) * WORD_SIZE;
    int words_per_row = padded_cols / WORD_SIZE;

    // Allocate host memory
    WORD *h_life = (WORD *)malloc(total_rows * words_per_row * sizeof(WORD));
    if (h_life == NULL)
    {
        fprintf(stderr, "Error allocating memory for h_life\n");
        exit(EXIT_FAILURE);
    }
    memset(h_life, 0, total_rows * words_per_row * sizeof(WORD));

    // Initialize the game board with random live/dead cells
    for (i = 1; i <= N; i++)
    {
        srand(54321 | i);
        for (j = 1; j <= N; j++)
        {
            int idx = i * words_per_row + ((j - 1) / WORD_SIZE);
            int bit = (j - 1) % WORD_SIZE;
            if (drand48() < 0.5)
                h_life[idx] |= ((WORD)1 << bit); // Set bit to 1 (ALIVE)
            else
                h_life[idx] &= ~((WORD)1 << bit); // Set bit to 0 (DIES)
        }
    }

    // Boundaries are already initialized to DIES (0) due to memset

    // Allocate device memory
    WORD *d_life, *d_temp;
    cudaError_t err;
    err = cudaMalloc((void **)&d_life, total_rows * words_per_row * sizeof(WORD));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: Failed to allocate device memory for d_life: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_temp, total_rows * words_per_row * sizeof(WORD));
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
    err = cudaMemcpy(d_life, h_life, total_rows * words_per_row * sizeof(WORD), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: Failed to copy data to d_life: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Define block and grid sizes
    int THREADS_X = 16;
    int THREADS_Y = 16;
    dim3 threadsPerBlock(THREADS_X, THREADS_Y);

    int numBlocksX = (words_per_row + THREADS_X - 1) / THREADS_X;
    int numBlocksY = (N + THREADS_Y - 1) / THREADS_Y;
    dim3 numBlocks(numBlocksX, numBlocksY);

    // Calculate shared memory size
    size_t shared_mem_size = (THREADS_Y + 2) * (THREADS_X + 2) * sizeof(WORD);

    // Start the timer
    t1 = gettime();

    int h_flag = 1;
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
        compute_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(d_life, d_temp, N, words_per_row, d_flag);

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
        WORD *temp_ptr = d_life;
        d_life = d_temp;
        d_temp = temp_ptr;
    }

    // Stop the timer
    t2 = gettime();
    printf("Time taken %f seconds for %d iterations on a board size of %d x %d\n", t2 - t1, k, N, N);
    fprintf(outfile, "Time taken %f seconds for %d iterations\n", t2 - t1, k);

    // Copy the result back to host
    err = cudaMemcpy(h_life, d_life, total_rows * words_per_row * sizeof(WORD), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: Failed to copy data from d_life to host: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Print final state to output file
    printarray(h_life, N, N, words_per_row, k, outfile);

    // Close the output file
    fclose(outfile);

    // Free memory
    free(h_life);
    cudaFree(d_life);
    cudaFree(d_temp);
    cudaFree(d_flag);

    return 0;
}