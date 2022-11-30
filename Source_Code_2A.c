#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void multisort(int*, int);

int main(int argc, char** argv) {
    int n = -1;
    int* array;
    int num_threads = -1;

    // Give number of elements
    do {
        printf("Give number of elements (N): ");
        scanf("%d", &n);
    } while(n < 0 ? printf("The number cannot be negative\n") : 0);

    // Initialize the array
    array=(int*)malloc(n*sizeof(int));

    // Fill the array with given values
    printf("Give %d values:\n", n);
    for(int i=0; i<n; i++) {
        scanf("%d", &array[i]);
    }

    // Give number of threads
    do {
        printf("Give number of threads (T): ");
        scanf("%d", &num_threads);
    } while(num_threads < 0 ? printf("The number cannot be negative\n") : 0);

    // Set number of threads
    omp_set_num_threads(num_threads);

    // Set up timer
    double start_time = omp_get_wtime();

    // Sort the array using recursive multisort
    #pragma omp parallel
    {
        #pragma omp single
        multisort(array, n);
    }

    // Print the sorted array
    printf("\nSorted\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }
    printf("\n"); 

    free(array);

    // Calculate time
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    printf("Elapsed time: %f seconds\n", elapsed_time);

    return 0;
}

// Recursive multisort function
void multisort(int *array, int n) {
    // If the array has 0 or 1 element, it is already sorted
    if (n <= 1) return;

    // Split the array in half
    int temp = n / 2;

    // Sort the left half in a separate task
    #pragma omp task
    multisort(array, temp);

    // Sort the right half in a separate task
    #pragma omp task
    multisort(array + temp, n - temp);

    // Wait for both tasks to finish
    #pragma omp taskwait

    // Merge the two sorted halves
    // Temporary array to store the merged result
    int *tempArray = (int*)malloc(n * sizeof(int));  

    // Indexes for the left half, right half, and merged array
    int i = 0, j = temp, k = 0;  

    // Merge the two halves in sorted order
    while (i < temp && j < n) { 
        if (array[i] < array[j]) {
            tempArray[k++] = array[i++];
        } else {
            tempArray[k++] = array[j++];
        }
    }
    
    // Copy the remaining elements from the left half
    while (i < temp) {  
        tempArray[k++] = array[i++];
    }

    // Copy the remaining elements from the right half
    while (j < n) {  
        tempArray[k++] = array[j++];
    }

    // Copy the merged array back to the original array
    for (i = 0; i < n; i++) {  
        array[i] = tempArray[i];
    }

    // Free the temporary array
    free(tempArray);  
}
