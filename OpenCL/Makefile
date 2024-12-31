CC = gcc
CFLAGS = -O1 
TARGET = dijkstra_opencl
SRC = dijkstra_opencl.c

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lOpenCL

clean:
	rm -f $(TARGET)
