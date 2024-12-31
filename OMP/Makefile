CC = gcc
CFLAGS = -O1 -fopenmp
TARGET = dijkstra_openmp
SRC = dijkstra_openmp.c

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
