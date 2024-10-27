#include <stdio.h>

int main() {
    // Times in microseconds
    float times[] = {2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9};
    int num_threads = 8;
    float max_time = 0.0;
    float total_execution_time = 0.0;
    float waiting_time = 0.0;

    // Calculate max time and total execution time
    for (int i = 0; i < num_threads; i++) {
        if (times[i] > max_time) {
            max_time = times[i];
        }
        total_execution_time += times[i];
    }

    // Calculate waiting time
    waiting_time = max_time * num_threads - total_execution_time;

    // Calculate percentage
    float waiting_percentage = (waiting_time / (max_time * num_threads)) * 100;
    printf("Percentage of threads' total execution time spent waiting for the barrier = %.1f%%\n", waiting_percentage);

    return 0;
}
