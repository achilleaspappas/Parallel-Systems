#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>

int main(int argc, char *argv[])
{
    
    int thread_num; //Number of threads
    int tid; // ID of thread
    int array_size; //Size of the array;
    int **array; // Array A
    int **arrayTwo; // Array B
    int diag_max = INT_MIN; // Max diagonal value
    int SDD_flag=0; // strictly diagonaly dominant flag
    int min_arrayTwo = INT_MAX;; // Min value for arrayTwo
    double timer1, timer2; // Variables to calculate execution time
    omp_lock_t lock; // Lock 

    printf("====================\n");
    printf("Initialazing...\n");
    printf("====================\n");

    // User inputs number of thread ans array size
    printf("Number of threads to use: ");
    scanf("%d", &thread_num);
    printf("Array size (NxN): ");
    scanf("%d", &array_size);

    // Creation of 2D array
    array = malloc(sizeof(int *)*array_size);
    for(int i=0; i<array_size; i++) {
        array[i] = malloc(sizeof(int *)*array_size);
    }
    /*
    if(array == NULL) {
        printf("Error occured while creating the array\n");
        exit(1);
    }*/

    // User inputs array elements
    printf("Insert array elements:\n");
    for(int i=0; i<array_size; i++) {
        for (int j=0; j<array_size; j++) {
            scanf("%d", &array[i][j]);
        }
    }

    // Print the current 2D array
    printf("\n2D Array:\n");
    for(int i=0; i<array_size; i++) {
        for (int j=0; j<array_size; j++) {
            printf("%d\t", array[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    // Setup lock
    omp_init_lock(&lock);

    // Set number of threads
    omp_set_num_threads(thread_num);

    // Start timer
    timer1 = omp_get_wtime();

    // Parallel region
    // The default for all values is shared except tid which is private
    #pragma omp parallel default(shared) private(tid) 
    {
        tid = omp_get_thread_num();

        /* Task A */
        #pragma omp master 
        {
            printf("====================\n");
            printf("Task A...\n");
            printf("====================\n");
        }

        // for loop to find sum and diagonal values
        #pragma omp for
        for(int i=0; i<array_size; i++) 
        {
            int sum = 0;
            int diag = 0;

            for(int j=0; j<array_size; j++) {
                if(i!=j) {
                    sum += abs(array[i][j]);
                } 
                else {
                    diag = abs(array[i][j]);
                }
            }

            // if a diagonal value is not bigger than the sum, we set the flag
            if(!(diag>sum)) {

                #pragma omp critical(SDD_flag)
                    SDD_flag = 1;
            }

        }

        // Checks the flag and prints the message
        #pragma omp single 
        {
            if(SDD_flag==1) {
                printf("2D array is NOT strictly diagonaly dominant.\n\n");
                exit(1);
            } else {
                printf("2D array is strictly diagonaly dominant.\n\n");
            }
        }


        /* Task B */
        #pragma omp master 
        {
            printf("====================\n");
            printf("Task B...\n");
            printf("====================\n");
        }

        // Finds the max diagonal value
        #pragma omp for reduction(max: diag_max)
        for(int i=1; i<array_size; i++) 
        {
            for(int j=0; j<array_size; j++) {
                if(i==j){
                    if(diag_max<abs(array[i][j])) {
                        diag_max = array[i][j];
                    }
                }
            }
        }

        // Prints the max diagonal value.
        #pragma omp single 
        {
            printf("Max diagonal value is: %d.\n\n", diag_max);
        }


        /* Task C */
        #pragma omp master 
        {
            printf("====================\n");
            printf("Task C...\n");
            printf("====================\n");
        }

        // Creation of new 2D array
        #pragma omp single 
        {
            arrayTwo = malloc(sizeof(int*)*array_size);
            for(int i=0; i<array_size; i++) {
                arrayTwo[i] = malloc(sizeof(int*)*array_size);
            }
        }

        // Calculate values to fill the new 2D array
        #pragma omp for 
        for(int i=0; i<array_size; i++) 
        {
            for(int j=0; j < array_size; j++) {
                if(i!=j) { 
                    arrayTwo[i][j] = diag_max - abs(array[i][j]);
                }
                else { 
                    arrayTwo[i][j] = diag_max;
                }
            }        
        }

        // Prints the new 2D array
        #pragma omp single
        {
            printf("New 2D Array:\n");
            for(int i=0; i<array_size; i++) {
                for (int j=0; j<array_size; j++) {
                    printf("%d\t", arrayTwo[i][j]);
                }
                printf("\n");
            }
        }

        /* Task D1 */
        #pragma omp master 
        {
            printf("====================\n");
            printf("Task D1...\n");
            printf("====================\n");
        }

        // Calculate min value of arrayTwo with reduction
        #pragma omp for reduction(min: min_arrayTwo)
        for(int i=0; i<array_size; i++) 
        { 
            for(int j=0; j<array_size; j++) {
                if(min_arrayTwo>arrayTwo[i][j]) { 
                    min_arrayTwo=arrayTwo[i][j];
                }
            }
        }

        #pragma omp single
        { 
            printf("Min value: %d\n\n", min_arrayTwo);
        }

        /* Task D2 */
        #pragma omp master 
        {
            printf("====================\n");
            printf("Task D2...\n");
            printf("====================\n");
        }

        // Calculate min value of arrayTwo with lock
        #pragma omp for
        for(int i=0; i<array_size; i++)
        { 
            for(int j = 0; j<array_size; j++) {
               
                if(min_arrayTwo > arrayTwo[i][j])
                    { 
                        omp_set_lock(&lock);
                        min_arrayTwo = arrayTwo[i][j];
                        omp_unset_lock(&lock);
                    }
                
            }
        }

        #pragma omp master
        { 
            printf("Min value: %d\n\n", min_arrayTwo);
        }

        /* Ending */
        #pragma omp master 
        {
            printf("====================\n");
            printf("Ending...\n");
            printf("====================\n");
        }

    }

    // End timer
  	timer2 = omp_get_wtime();

    printf("Time = %f \n", timer2-timer1);

    // Free memory
    for(int i=0; i<array_size; i++) {
        free(array[i]);
    }
    free(array);

    for(int i=0; i<array_size; i++) {
        free(arrayTwo[i]);
    }
    free(arrayTwo);
}
