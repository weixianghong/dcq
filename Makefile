all:
	g++ -o demo demo.cpp util/array_io_txt.cpp algorithm/train.cpp algorithm/args_for_compute_gradient.cpp -L/usr/local/lib -lopencv_core -llbfgs -fopenmp

