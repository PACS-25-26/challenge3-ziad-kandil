CXX = mpicxx

CXXFLAGS = -O3 -std=c++20 -fopenmp -IInclude

SRC = $(wildcard Src/*.cpp)

TARGET = main

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: 
	mpirun -np 2 ./$(TARGET)