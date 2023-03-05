#include "array_io_txt.h"

int** read_from_txt(const char* input, int num_sample, int num_dimension){

	std::ifstream infile(input);
	std::string line;

	int	N = num_sample;
    int	D = num_dimension;

	int** learn = new int*[N];
	for(int i = 0; i < N; ++i) {
	    learn[i] = new int[D];
	}

	int i = 0, j = 0;
	
	while (std::getline(infile, line)) {
		std::istringstream iss(line);
		int temp;

		j = 0;
		while (iss >> temp) {
			learn[j][i] = temp;
			j++;
		}
	i++;
	}

	return learn;

}
