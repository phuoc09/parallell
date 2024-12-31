#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>

#define INFNTY INT_MAX
#define THREADS_PER_BLOCK 256

// Kernel tìm đỉnh có giá trị nhỏ nhất chưa thăm
__global__ void find_min_vertex(int *distances, int *visited, int *min_vertex, int V) {
    __shared__ int local_min_dist[THREADS_PER_BLOCK];
    __shared__ int local_min_vertex[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < V && !visited[idx]) {
        local_min_dist[tid] = distances[idx];
        local_min_vertex[tid] = idx;
    } else {
        local_min_dist[tid] = INFNTY;
        local_min_vertex[tid] = -1;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (local_min_dist[tid + stride] < local_min_dist[tid]) {
                local_min_dist[tid] = local_min_dist[tid + stride];
                local_min_vertex[tid] = local_min_vertex[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(min_vertex, local_min_vertex[0]);
    }
}

// Kernel cập nhật khoảng cách
__global__ void relax_edges(int *adj_matrix, int *distances, int *visited, int current_vertex, int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v < V && !visited[v]) {
        int weight = adj_matrix[current_vertex * V + v];
        if (weight != INFNTY && distances[current_vertex] != INFNTY) {
            int new_distance = distances[current_vertex] + weight;
            atomicMin(&distances[v], new_distance);
        }
    }
}

// Hàm Dijkstra hoàn toàn trên GPU
void dijkstra_cuda_full_gpu(int V, int *h_adj_matrix, int source, int *h_distances) {
    // Cấp phát bộ nhớ trên GPU
    int *d_adj_matrix, *d_distances, *d_visited, *d_min_vertex;
    cudaMalloc(&d_adj_matrix, V * V * sizeof(int));
    cudaMalloc(&d_distances, V * sizeof(int));
    cudaMalloc(&d_visited, V * sizeof(int));
    cudaMalloc(&d_min_vertex, sizeof(int));

    // Tính thời gian
    cudaEvent_t start, stop;
    float time_upload = 0, time_compute = 0, time_download = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // **Thời gian nạp dữ liệu**
    cudaEventRecord(start);
    cudaMemcpy(d_adj_matrix, h_adj_matrix, V * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_visited, 0, V * sizeof(int));
    int *h_init_distances = (int *)malloc(V * sizeof(int));
    for (int i = 0; i < V; i++) h_init_distances[i] = INFNTY;
    h_init_distances[source] = 0;
    cudaMemcpy(d_distances, h_init_distances, V * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_upload, start, stop);
    free(h_init_distances);

    // **Thời gian tính toán**
    cudaEventRecord(start);
    int num_blocks = (V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    for (int count = 0; count < V - 1; count++) {
        int h_min_vertex = -1;
        cudaMemcpy(d_min_vertex, &h_min_vertex, sizeof(int), cudaMemcpyHostToDevice);
        find_min_vertex<<<num_blocks, THREADS_PER_BLOCK>>>(d_distances, d_visited, d_min_vertex, V);
        cudaMemcpy(&h_min_vertex, d_min_vertex, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_min_vertex == -1) break;
        int visited = 1;
        cudaMemcpy(&d_visited[h_min_vertex], &visited, sizeof(int), cudaMemcpyHostToDevice);
        relax_edges<<<num_blocks, THREADS_PER_BLOCK>>>(d_adj_matrix, d_distances, d_visited, h_min_vertex, V);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_compute, start, stop);

    // **Thời gian tải kết quả**
    cudaEventRecord(start);
    cudaMemcpy(h_distances, d_distances, V * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_download, start, stop);

    // Giải phóng bộ nhớ
    cudaFree(d_adj_matrix);
    cudaFree(d_distances);
    cudaFree(d_visited);
    cudaFree(d_min_vertex);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Hiển thị thời gian
    printf("Upload Time (CPU to GPU): %f ms\n", time_upload);
    printf("Computation Time (GPU): %f ms\n", time_compute);
    printf("Download Time (GPU to CPU): %f ms\n", time_download);
    printf("Full Time Parallel: %f ms\n", time_download+time_compute+time_upload);
}
void dijkstra(int V, int *adjacency_matrix, int source, int *distances) {
    int *visited = (int *)malloc(V * sizeof(int));
    
    for (int i = 0; i < V; i++) {
        distances[i] = INFNTY;
        visited[i] = 0;
    }
    distances[source] = 0;

    for (int count = 0; count < V - 1; count++) {
        // Tìm đỉnh chưa thăm với khoảng cách nhỏ nhất
        int min_dist = INFNTY, u = -1;
        for (int v = 0; v < V; v++) {
            if (!visited[v] && distances[v] <= min_dist) {
                min_dist = distances[v];
                u = v;
            }
        }

        visited[u] = 1;

        // Cập nhật khoảng cách cho các đỉnh lân cận
        for (int v = 0; v < V; v++) {
            if (!visited[v] && adjacency_matrix[u * V + v] && distances[u] != INFNTY &&
                distances[u] + adjacency_matrix[u * V + v] < distances[v]) {
                distances[v] = distances[u] + adjacency_matrix[u * V + v];
            }
        }
    }

    free(visited);
}
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
int count_differences(int *array1, int *array2, int size) {
    int differences = 0;                                                                                                                                                                                                                        
    for (int i = 0; i < size; i++) {
        if (array1[i] != array2[i]) {
            differences++;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        }                   
    }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           differences=0;        
    return differences;
}
int main(int argc, char **argv) {
    if (argc != 2) {
        printf("USAGE: ./dijkstra_cuda <number_of_vertices>\n");
        return 1;
    }

    int V = atoi(argv[1]);
    int *adj_matrix = (int *)malloc(V * V * sizeof(int));
    int *distances_serial= (int *)malloc(V * sizeof(int));
    int *distances_parallel = (int *)malloc(V * sizeof(int));
    generate_random_graph(V, adj_matrix);

    clock_t start = clock();
    dijkstra(V, adj_matrix, 0, distances_serial); // Tìm đường đi từ đỉnh nguồn 0
    clock_t end = clock();
    printf("CPU Execution Time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    dijkstra_cuda_full_gpu(V, adj_matrix, 0, distances_parallel);

    printf("\n Different: %d\n", count_differences(distances_serial,distances_parallel,V));


    free(adj_matrix);
    free(distances_parallel);
    free(distances_serial);
    return 0;
}
