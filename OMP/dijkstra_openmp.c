#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#define INFNTY INT_MAX
#define THREADS 6
// Hàm tạo đồ thị ngẫu nhiên
void generate_random_graph(int V, int *matrix) {
    srand(time(NULL));

    for (int i = 0; i < V; i++) {
        int edges = rand() % 7; // Số đỉnh kề ngẫu nhiên từ 0 đến 6
        int count = 0;

        for (int j = i; j < V; j++) {
            if (i == j) {
                matrix[i * V + j] = 0; // Không có cạnh tự vòng
            } else if (count < edges && (rand() % (V - j)) < (edges - count)) {
                int weight = rand() % 10 + 1; // Trọng số từ 1 đến 10
                matrix[i * V + j] = weight;
                matrix[j * V + i] = weight; // Đảm bảo đối xứng
                count++;
            } else {
                matrix[i * V + j] = INT_MAX;
                matrix[j * V + i] = INT_MAX; // Đảm bảo đối xứng
            }
        }
    }
}

// Hàm in ma trận kề
void print_adjacency_matrix(int V, int *adjacency_matrix) {
    printf("\nADJACENCY MATRIX:\n");
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (adjacency_matrix[i * V + j] == INFNTY) {
                printf("INF ");
            } else {
                printf("%d ", adjacency_matrix[i * V + j]);
            }
        }
        printf("\n");
    }
}

// Thuật toán Dijkstra tuần tự
void dijkstra_serial(int V, int *adjacency_matrix, int source, int *distances) {
    int *visited = (int *)malloc(V * sizeof(int));

    for (int i = 0; i < V; i++) {
        distances[i] = INFNTY;
        visited[i] = 0;
    }
    distances[source] = 0;

    for (int count = 0; count < V - 1; count++) {
        int min_dist = INFNTY, u = -1;

        // Tìm đỉnh có khoảng cách nhỏ nhất chưa thăm
        for (int v = 0; v < V; v++) {
            if (!visited[v] && distances[v] <= min_dist) {
                min_dist = distances[v];
                u = v;
            }
        }

        visited[u] = 1;

        // Cập nhật khoảng cách cho các đỉnh lân cận
        for (int v = 0; v < V; v++) {
            if (!visited[v] && adjacency_matrix[u * V + v] &&
                distances[u] != INFNTY &&
                distances[u] + adjacency_matrix[u * V + v] < distances[v]) {
                distances[v] = distances[u] + adjacency_matrix[u * V + v];
            }
        }
    }

    free(visited);
}

// Thuật toán Dijkstra song song bằng OpenMP
void dijkstra_parallel(int V, int *adjacency_matrix, int source, int *distances) {


    int *visited = (int *)malloc(V * sizeof(int));
    #pragma omp parallel for
    for (int i = 0; i < V; i++) {
        distances[i] = INFNTY;
        visited[i] = 0;
    }
    distances[source] = 0;

    for (int count = 0; count < V - 1; count++) {
        int min_dist = INFNTY;
        int current_vertex = -1;

        // Find the vertex with the minimum distance that hasn't been visited (parallel section)
        #pragma omp parallel shared(min_dist, current_vertex, visited) 
        {
            int min_threads = INFNTY;
            int thread_vertex = -1;

            // Each thread works on a part of the array
            #pragma omp for
            for (int v = 0; v < V; v++) {
                if (!visited[v] && distances[v] < min_threads) {
                    min_threads = distances[v];
                    thread_vertex = v;
                }
            }

            // Update the global minimum distance and vertex in a critical section
            #pragma omp critical
            {
                if (min_threads < min_dist) {
                    min_dist = min_threads;
                    current_vertex = thread_vertex;
                }
            }
        }
        visited[current_vertex] = 1;

        // Cập nhật khoảng cách cho các đỉnh lân cận (song song)
        #pragma omp parallel for
        for (int v = 0; v < V; v++) {
            if (!visited[v] && adjacency_matrix[current_vertex * V + v] &&
                distances[current_vertex] != INFNTY &&
                distances[current_vertex] + adjacency_matrix[current_vertex * V + v] < distances[v]) {
                distances[v] = distances[current_vertex] + adjacency_matrix[current_vertex * V + v];
            }
        }
    }

    free(visited);
}

int count_differences(int *array1, int *array2, int size) {
    int differences = 0;
    for (int i = 0; i < size; i++) {
        if (array1[i] != array2[i]) {
            differences++;
        }
    }
    return differences;
}
int main(int argc, char **argv) {
    if (argc != 2) {
        printf("USAGE: ./dijkstra_omp <number_of_vertices>\n");
        return 1;
    }

    int V = atoi(argv[1]);
    int *adjacency_matrix = (int *)malloc(V * V * sizeof(int));
    int *distances_serial = (int *)malloc(V * sizeof(int));
    int *distances_parallel = (int *)malloc(V * sizeof(int));

    generate_random_graph(V, adjacency_matrix);


    // Chạy thuật toán tuần tự
    double start_serial = omp_get_wtime();
    dijkstra_serial(V, adjacency_matrix, 0, distances_serial);
    double end_serial = omp_get_wtime();
    omp_set_num_threads(THREADS);
    // Chạy thuật toán song song
    double start_parallel = omp_get_wtime();
    dijkstra_parallel(V, adjacency_matrix, 0, distances_parallel);
    double end_parallel = omp_get_wtime();
    

    printf("\nSerial Execution Time: %f seconds\n", end_serial - start_serial);
    printf("Parallel Execution Time: %f seconds\n", end_parallel - start_parallel);
    printf("\n Different: %d\n", count_differences(distances_serial,distances_parallel,V));
    free(adjacency_matrix);
    free(distances_serial);
    free(distances_parallel);

    return 0;
}
