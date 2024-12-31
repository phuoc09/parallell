CC = nvcc
CFLAGS = -O1
TARGET = dijkstra_cuda
SRC = dijkstra_cuda.cu

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
