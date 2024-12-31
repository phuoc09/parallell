#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <CL/cl.h>
#include <time.h>

#define INFNTY INT_MAX
#define MAX_SOURCE_SIZE 0x10000

void dijkstra_serial(int V, int *adjacency_matrix, int source, int *distances) {
    int *visited = (int *)malloc(V * sizeof(int));

    for (int i = 0; i < V; i++) {
        distances[i] = INFNTY;
        visited[i] = 0;
    }
    distances[source] = 0;

    for (int count = 0; count < V - 1; count++) {
        // Find the vertex with the minimum distance
        int min_dist = INFNTY, u = -1;
        for (int v = 0; v < V; v++) {
            if (!visited[v] && distances[v] <= min_dist) {
                min_dist = distances[v];
                u = v;
            }
        }

        visited[u] = 1;

        // Update distances for adjacent vertices
        for (int v = 0; v < V; v++) {
            if (!visited[v] && adjacency_matrix[u * V + v] != INFNTY && distances[u] != INFNTY &&
                distances[u] + adjacency_matrix[u * V + v] < distances[v]) {
                distances[v] = distances[u] + adjacency_matrix[u * V + v];
            }
        }
    }

    free(visited);
}

// OpenCL kernel for edge relaxation
const char *kernelSource = "__kernel void relax_edges(\n"
    "    __global int *adj_matrix,\n"
    "    __global int *distances,\n"
    "    __global int *visited,\n"
    "    int current_vertex,\n"
    "    int V) {\n"
    "    int v = get_global_id(0);\n"
    "    if (v < V && !visited[v]) {\n"
    "        int weight = adj_matrix[current_vertex * V + v];\n"
    "        if (weight != INT_MAX && distances[current_vertex] != INT_MAX) {\n"
    "            int new_distance = distances[current_vertex] + weight;\n"
    "            atomic_min(&distances[v], new_distance);\n"
    "        }\n"
    "    }\n"
    "}\n";

void dijkstra_opencl(int V, int *h_adj_matrix, int source, int *h_distances) {
    // Initialize OpenCL context, command queue, and program
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_int ret;

    ret = clGetPlatformIDs(1, &platform_id, NULL);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

    size_t source_size = strlen(kernelSource);
    program = clCreateProgramWithSource(context, 1, &kernelSource, &source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "relax_edges", &ret);

    // Allocate device memory and transfer data
    cl_mem d_adj_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY, V * V * sizeof(int), NULL, &ret);
    cl_mem d_distances = clCreateBuffer(context, CL_MEM_READ_WRITE, V * sizeof(int), NULL, &ret);
    cl_mem d_visited = clCreateBuffer(context, CL_MEM_READ_WRITE, V * sizeof(int), NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, d_adj_matrix, CL_TRUE, 0, V * V * sizeof(int), h_adj_matrix, 0, NULL, NULL);
    int *h_init_distances = (int *)malloc(V * sizeof(int));
    for (int i = 0; i < V; i++) h_init_distances[i] = INFNTY;
    h_init_distances[source] = 0;
    ret = clEnqueueWriteBuffer(command_queue, d_distances, CL_TRUE, 0, V * sizeof(int), h_init_distances, 0, NULL, NULL);

    int *h_visited = (int *)calloc(V, sizeof(int));
    ret = clEnqueueWriteBuffer(command_queue, d_visited, CL_TRUE, 0, V * sizeof(int), h_visited, 0, NULL, NULL);

    // Run the Dijkstra algorithm
    for (int count = 0; count < V - 1; count++) {
        int min_dist = INFNTY, current_vertex = -1;
        for (int v = 0; v < V; v++) {
            if (!h_visited[v] && h_init_distances[v] < min_dist) {
                min_dist = h_init_distances[v];
                current_vertex = v;
            }
        }

        if (current_vertex == -1) break;

        h_visited[current_vertex] = 1;
        ret = clEnqueueWriteBuffer(command_queue, d_visited, CL_TRUE, 0, V * sizeof(int), h_visited, 0, NULL, NULL);

        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_adj_matrix);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_distances);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_visited);
        ret = clSetKernelArg(kernel, 3, sizeof(int), &current_vertex);
        ret = clSetKernelArg(kernel, 4, sizeof(int), &V);

        size_t global_work_size = V;
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue, d_distances, CL_TRUE, 0, V * sizeof(int), h_init_distances, 0, NULL, NULL);
    }

    memcpy(h_distances, h_init_distances, V * sizeof(int));

    // Clean up
    clReleaseMemObject(d_adj_matrix);
    clReleaseMemObject(d_distances);
    clReleaseMemObject(d_visited);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(h_visited);
    free(h_init_distances);
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
        printf("USAGE: ./dijkstra_opencl <number_of_vertices>\n");
        return 1;
    }

    int V = atoi(argv[1]);
    int *adj_matrix = (int *)malloc(V * V * sizeof(int));
    int *distances_serial = (int *)malloc(V * sizeof(int));
    int *distances_parallel = (int *)malloc(V * sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            adj_matrix[i * V + j] = (i == j) ? 0 : (rand() % 10 + 1);
        }
    }

    clock_t start = clock();
    dijkstra_serial(V, adj_matrix, 0, distances_serial);
    clock_t end = clock();
    printf("Serial Execution Time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
    dijkstra_opencl(V, adj_matrix, 0, distances_parallel);
    end = clock();
    printf("OpenCL Execution Time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    printf("\n Different: %d\n", count_differences(distances_serial,distances_parallel,V));
    free(adj_matrix);
    free(distances_serial);
    free(distances_parallel);

    return 0;
}
