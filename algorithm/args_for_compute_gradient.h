#include <stdio.h>
#include <map>
#include <cmath>
#include <string>
#include <stdlib.h>
#include <iostream>

#include "/usr/local/include/lbfgs.h"

using namespace std;

class args_for_compute_gradient{
	public:
		int N;
		int M;
		int K;
		int D;

		double epsilon;
		int** B;
		int** training_set;
		map<string, double> parameters;

		args_for_compute_gradient(int N__, const int M__, int D__, int K__);
		~args_for_compute_gradient();

		void SET_epsilon(double* p_epsilon);
		void SET_B(int** B__);
		void SET_training_set(int** training_set__);
		void SET_parameters(map<string, double> parameters__);

};
