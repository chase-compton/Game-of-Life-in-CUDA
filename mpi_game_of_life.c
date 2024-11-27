/*
    Name: Chase Compton
    Email: cscompton1@crimson.ua.edu
    Course Section: CS 481
    Homework #4: MPI Game of Life

    Implements an efficient message-passing version of the Game of Life using MPI.

    Instructions to compile the program: mpicc -O3 -o hw4 hw4.c -lm

    Instructions to run the program: mpirun -np <num_procs> ./hw4 <size> <max_iterations> <num_processes> <output_directory>
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <mpi.h>

#define DIES 0
#define ALIVE 1

// Function to get current time in seconds
double gettime(void)
{
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return ((double)tval.tv_sec + (double)tval.tv_usec / 1000000.0);
}

// Function to allocate a 2D array dynamically
int **allocarray(int P, int Q)
{
    int i, *p, **a;
    p = (int *)malloc(P * Q * sizeof(int));
    a = (int **)malloc(P * sizeof(int *));
    for (i = 0; i < P; i++)
        a[i] = &p[i * Q];
    return a;
}

// Function to free the dynamically allocated 2D array
void freearray(int **a)
{
    free(&a[0][0]);
    free(a);
}

// Function to print the current state of the Game of Life board
void printarray(int **a, int nRows, int nCols, int k, FILE *outfile)
{
    int i, j;
    char *border = malloc(sizeof(char) * (nCols * 2 + 4));
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

// Function to compute the next generation of the Game of Life
int compute(int **life, int **temp, int nRows, int nCols)
{
    int i, j, value, flag = 0;

    for (i = 1; i <= nRows; i++)
    {
        for (j = 1; j <= nCols; j++)
        {
            value = life[i - 1][j - 1] + life[i - 1][j] + life[i - 1][j + 1] +
                    life[i][j - 1] + life[i][j + 1] +
                    life[i + 1][j - 1] + life[i + 1][j] + life[i + 1][j + 1];

            if (life[i][j])
            {
                if (value < 2 || value > 3)
                {
                    temp[i][j] = DIES;
                    flag++;
                }
                else
                {
                    temp[i][j] = ALIVE;
                }
            }
            else
            {
                if (value == 3)
                {
                    temp[i][j] = ALIVE;
                    flag++;
                }
                else
                {
                    temp[i][j] = DIES;
                }
            }
        }
    }
    return flag;
}

int main(int argc, char **argv)
{
    int N, NTIMES, num_procs, cur_rank;
    int i, j, k, flag = 1;
    int **life = NULL, **temp = NULL, **ptr;
    double t1, t2;
    char *output_dir;
    FILE *outfile = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &cur_rank);

    if (argc != 5)
    {
        if (cur_rank == 0)
            printf("Usage: mpirun -np <num_procs> %s <size> <max_iterations> <num_processes> <output_directory>\n", argv[0]);
        MPI_Finalize();
        exit(-1);
    }

    N = atoi(argv[1]);
    NTIMES = atoi(argv[2]);
    int num_processes = atoi(argv[3]);
    output_dir = argv[4];

    if (num_processes != num_procs)
    {
        if (cur_rank == 0)
            printf("Error: Number of processes specified does not match number of processes started.\n");
        MPI_Finalize();
        exit(-1);
    }

    // Only process 0 opens the output file
    if (cur_rank == 0)
    {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "%s/%d_%d_%d_output.txt", output_dir, N, NTIMES, num_procs);
        outfile = fopen(output_filename, "w");
        if (outfile == NULL)
        {
            printf("Error opening output file\n");
            MPI_Finalize();
            exit(-1);
        }
    }

    // Compute local dimensions
    int rows_per_proc = N / num_procs;
    int extra_rows = N % num_procs;
    int nRows = rows_per_proc + (cur_rank < extra_rows ? 1 : 0);
    int start_row = cur_rank * rows_per_proc + (cur_rank < extra_rows ? cur_rank : extra_rows) + 1;

    // Allocate memory for the local game boards with ghost cells
    life = allocarray(nRows + 2, N + 2);
    temp = allocarray(nRows + 2, N + 2);

    // Initialize the boundaries of the life board
    for (i = 0; i < nRows + 2; i++)
    {
        for (j = 0; j < N + 2; j++)
        {
            life[i][j] = DIES;
            temp[i][j] = DIES;
        }
    }

    // Prepare for scattering the initial board
    int *recvcounts = NULL;
    int *displs = NULL;
    int *flat_life = (int *)malloc(nRows * N * sizeof(int));
    int *full_life_flat = NULL;

    if (cur_rank == 0)
    {
        // Allocate the full board
        int **full_life = allocarray(N + 2, N + 2);
        // Initialize boundaries
        for (i = 0; i < N + 2; i++)
            for (j = 0; j < N + 2; j++)
                full_life[i][j] = DIES;

        // Initialize the game board with random live/dead cells
        for (i = 1; i <= N; i++)
        {
            srand(54321 | i);
            for (j = 1; j <= N; j++)
            {
                if (drand48() < 0.5)
                    full_life[i][j] = ALIVE;
                else
                    full_life[i][j] = DIES;
            }
        }

        // Flatten the data into a 1D array
        full_life_flat = (int *)malloc(N * N * sizeof(int));
        int idx = 0;
        for (i = 1; i <= N; i++)
        {
            for (j = 1; j <= N; j++)
            {
                full_life_flat[idx++] = full_life[i][j];
            }
        }

        // Prepare counts and displacements for Scatterv
        recvcounts = (int *)malloc(num_procs * sizeof(int));
        displs = (int *)malloc(num_procs * sizeof(int));
        int offset = 0;
        for (i = 0; i < num_procs; i++)
        {
            int rows = rows_per_proc + (i < extra_rows ? 1 : 0);
            recvcounts[i] = rows * N;
            displs[i] = offset;
            offset += recvcounts[i];
        }

        freearray(full_life);
    }

    // Scatter the initial board to all processes
    MPI_Scatterv(full_life_flat, recvcounts, displs, MPI_INT,
                 flat_life, nRows * N, MPI_INT, 0, comm);

    // Copy the flat_life into the life array
    int idx = 0;
    for (i = 1; i <= nRows; i++)
    {
        for (j = 1; j <= N; j++)
        {
            life[i][j] = flat_life[idx++];
        }
    }

    if (cur_rank == 0)
    {
        free(full_life_flat);
        free(recvcounts);
        free(displs);
    }
    free(flat_life);

    // Start the timer
    if (cur_rank == 0)
        t1 = gettime();

    MPI_Status status;
    int up = cur_rank - 1;
    int down = cur_rank + 1;
    if (up < 0)
        up = MPI_PROC_NULL;
    if (down >= num_procs)
        down = MPI_PROC_NULL;

    for (k = 0; k < NTIMES && flag != 0; k++)
    {
        int local_changes = 0;

        // Send and receive rows to/from neighboring processes
        MPI_Sendrecv(life[1], N + 2, MPI_INT, up, 0,
                     life[nRows + 1], N + 2, MPI_INT, down, 0,
                     comm, &status);

        MPI_Sendrecv(life[nRows], N + 2, MPI_INT, down, 1,
                     life[0], N + 2, MPI_INT, up, 1,
                     comm, &status);

        // Compute the next generation
        local_changes = compute(life, temp, nRows, N);

        // Check for changes across all processes
        MPI_Allreduce(&local_changes, &flag, 1, MPI_INT, MPI_SUM, comm);

        // Swap life and temp
        ptr = life;
        life = temp;
        temp = ptr;
    }

    // Stop the timer
    if (cur_rank == 0)
    {
        t2 = gettime();
        printf("Time taken %f seconds for %d iterations on a board size of %d x %d\n", t2 - t1, k, N, N);
        fprintf(outfile, "Time taken %f seconds for %d iterations\n", t2 - t1, k);
    }

    // Gather final state to process 0
    int *sendcounts = NULL;
    int *sdispls = NULL;
    int *gather_flat_life = (int *)malloc(nRows * N * sizeof(int));

    // Flatten the local life array into gather_flat_life
    idx = 0;
    for (i = 1; i <= nRows; i++)
    {
        for (j = 1; j <= N; j++)
        {
            gather_flat_life[idx++] = life[i][j];
        }
    }

    if (cur_rank == 0)
    {
        sendcounts = (int *)malloc(num_procs * sizeof(int));
        sdispls = (int *)malloc(num_procs * sizeof(int));
        int offset = 0;
        for (i = 0; i < num_procs; i++)
        {
            int rows = rows_per_proc + (i < extra_rows ? 1 : 0);
            sendcounts[i] = rows * N;
            sdispls[i] = offset;
            offset += sendcounts[i];
        }
    }

    int *final_life = NULL;
    if (cur_rank == 0)
    {
        final_life = (int *)malloc(N * N * sizeof(int));
    }

    MPI_Gatherv(gather_flat_life, nRows * N, MPI_INT,
                final_life, sendcounts, sdispls, MPI_INT, 0, comm);

    if (cur_rank == 0)
    {
        // Convert flat array to 2D array for printing
        int **full_life = allocarray(N + 2, N + 2);

        // Initialize boundaries
        for (i = 0; i < N + 2; i++)
            for (j = 0; j < N + 2; j++)
                full_life[i][j] = DIES;

        idx = 0;
        for (i = 1; i <= N; i++)
            for (j = 1; j <= N; j++)
                full_life[i][j] = final_life[idx++];

        // Print final state to output file
        printarray(full_life, N, N, k, outfile);

        freearray(full_life);
        fclose(outfile);
        free(final_life);
        free(sendcounts);
        free(sdispls);
    }

    // Free allocated memory
    freearray(life);
    freearray(temp);
    free(gather_flat_life);

    if (cur_rank == 0)
        printf("Program terminates normally\n");

    MPI_Finalize();
    return 0;
}