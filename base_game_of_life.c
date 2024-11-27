/*
    Name: Chase Compton
    Email: cscompton1@crimson.ua.edu
    Course Section: CS 481
    Homework Template
    Modified from the provided life.c file.
    Implements the base Game of Life algorithm with the additional requirements.

*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <_stdio.h>

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
void printarray(int **a, int N, int k, FILE *outfile)
{
    int i, j;
    char *border = malloc(sizeof(char) * (N * 2 + 4));
    memset(border, '-', N * 2 + 3);
    border[N * 2 + 3] = '\0';

    if (outfile == stdout)
    {
        printf("Life after %d iterations:\n", k);
    }
    else
    {
        fprintf(outfile, "Final state after %d iterations:\n", k);
    }

    fprintf(outfile, "%s\n", border);
    for (i = 1; i < N + 1; i++)
    {
        fprintf(outfile, "| ");
        for (j = 1; j < N + 1; j++)
            fprintf(outfile, "%s ", a[i][j] ? "■" : "□");
        fprintf(outfile, "|\n");
    }
    fprintf(outfile, "%s\n\n", border);

    free(border);
}

// Function to compute the next generation of the Game of Life
int compute(int **life, int **temp, int N)
{
    int i, j, value, flag = 0;

    for (i = 1; i < N + 1; i++)
    {
        for (j = 1; j < N + 1; j++)
        {
            value = life[i - 1][j - 1] + life[i - 1][j] +
                    life[i - 1][j + 1] + life[i][j - 1] +
                    life[i][j + 1] + life[i + 1][j - 1] +
                    life[i + 1][j] + life[i + 1][j + 1];

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
    int N, NTIMES, **life = NULL, **temp = NULL, **ptr;
    int i, j, k, flag = 1;
    double t1, t2;
    char *output_dir;
    FILE *outfile;

    if (argc != 4)
    {
        printf("Usage: %s <size> <max. iterations> <output_directory>\n", argv[0]);
        exit(-1);
    }

    N = atoi(argv[1]);
    NTIMES = atoi(argv[2]);
    output_dir = argv[3];

    // Open output file
    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "%s/%d_%d_output.txt",
             output_dir, N, NTIMES);
    outfile = fopen(output_filename, "w");
    if (outfile == NULL)
    {
        printf("Error opening output file\n");
        exit(-1);
    }

    // Allocate memory for the game boards
    life = allocarray(N + 2, N + 2);
    temp = allocarray(N + 2, N + 2);

    // Initialize the boundaries of the life board
    for (i = 0; i < N + 2; i++)
    {
        life[0][i] = life[i][0] = life[N + 1][i] = life[i][N + 1] = DIES;
        temp[0][i] = temp[i][0] = temp[N + 1][i] = temp[i][N + 1] = DIES;
    }

    // Initialize the game board with random live/dead cells
    for (i = 1; i < N + 1; i++)
    {
        srand(54321 | i);
        for (j = 1; j < N + 1; j++)
            life[i][j] = (drand48() < 0.5) ? ALIVE : DIES;
    }

// Print initial state if DEBUG1 is defined
#ifdef DEBUG1
    printarray(life, N, 0, stdout);
#endif
    // Start the timer
    t1 = gettime();
    // Main game loop
    for (k = 0; k < NTIMES && flag != 0; k++)
    {
        flag = compute(life, temp, N);
        ptr = life;
        life = temp;
        temp = ptr;
// Print debug information if DEBUG2 is defined
#ifdef DEBUG2
        printf("No. of cells whose value changed in iteration %d = %d\n", k + 1, flag);
        printarray(life, N, k + 1, stdout);
#endif
    }
    // Stop the timer
    t2 = gettime();

    printf("Time taken %f seconds for %d iterations on a board size of %d x %d\n", t2 - t1, k, N, N);
    fprintf(outfile, "Time taken %f seconds for %d iterations\n", t2 - t1, k);
    printarray(life, N, k, outfile);

    freearray(life);
    freearray(temp);
    fclose(outfile);

    printf("Program terminates normally\n");

    return 0;
}