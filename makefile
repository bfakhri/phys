engine: phys.cpp main.cpp mather.h sphere.cpp shape.cpp sphere.h phys.h shape.h cube.h cube.cpp mather.cpp capture.h capture.cpp
	g++ -o engine.exe phys.cpp main.cpp sphere.cpp shape.cpp cube.cpp mather.cpp capture.cpp -lGL -lGLU -lglut -std=c++11 -g -fopenmp -lpng

clean:
	rm *.exe
	echo Done Cleaning.
