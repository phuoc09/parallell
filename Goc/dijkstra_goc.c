#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#define TRUE 1
#define FALSE 0
#define INFNTY INT_MAX
typedef int boolean;

/* Generates a random undirected graph represented by an adjacency matrix */
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
                matrix[i * V + j] = 0;
                matrix[j * V + i] = 0; // Đảm bảo đối xứng
            }
        }
    }
}

/* Print adjacency matrix */
void print_adjacency_matrix(int V, int *adjacency_matrix)
{
    printf("\nADJACENCY MATRIX:\n");
    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V; j++)
        {
            printf("%d ", adjacency_matrix[i * V + j]);
        }
        printf("\n");
    }
}

/* Finds vertex with the minimum distance among the vertices that have not been visited yet */
int find_min_distance(int V, int *distance, boolean *visited)
{
    int min_distance = INFNTY;
    int min_index = -1;

    for (int v = 0; v < V; v++)
    {
        if (!visited[v] && distance[v] <= min_distance)
        {
            min_distance = distance[v];
            min_index = v;
        }
    }
    return min_index;
}



void dijkstra_serial(int V, int *adjacency_matrix, int *len, int *temp_distance)
{
    boolean *visited = (boolean *)malloc(V * sizeof(boolean));
    clock_t start = clock();

        int source =0;
        for (int i = 0; i < V; i++)
        {
            visited[i] = FALSE;
            temp_distance[i] = INFNTY;
            len[source * V + i] = INFNTY;
        }

        len[source * V + source] = 0;

        for (int count = 0; count < V - 1; count++)
        {
            int current_vertex = find_min_distance(V, len + source * V, visited);
            visited[current_vertex] = TRUE;

            for (int v = 0; v < V; v++)
            {
                int weight = adjacency_matrix[current_vertex * V + v];
                if (!visited[v] && weight && len[source * V + current_vertex] != INFNTY &&
                    len[source * V + current_vertex] + weight < len[source * V + v])
                {
                    len[source * V + v] = len[source * V + current_vertex] + weight;
                    temp_distance[v] = len[source * V + v];
                }
            }
        }

    clock_t end = clock();
    printf("TOTAL ELAPSED TIME ON CPU = %f SECS\n", (float)(end - start) / CLOCKS_PER_SEC);
    free(visited);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("USAGE: ./dijkstra_serial <number_of_vertices>\n");
        return 1;
    }

    int *len_serial, *temp_distance;
    int V = atoi(argv[1]);

    len_serial = (int *)malloc(V * V * sizeof(int));
    temp_distance = (int *)malloc(V * sizeof(int));

    int *adjacency_matrix = (int *)malloc(V * V * sizeof(int));

    generate_random_graph(V, adjacency_matrix);
    dijkstra_serial(V, adjacency_matrix, len_serial, temp_distance);
    free(len_serial);
    free(temp_distance);
    free(adjacency_matrix);

    return 0;
}
