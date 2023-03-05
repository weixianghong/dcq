#include "args_for_compute_gradient.h"

args_for_compute_gradient::args_for_compute_gradient(int N__, const int M__, int D__, int K__, int num_sep__){
	N = N__;
	M = M__;
	D = D__;
	K = K__;
	num_sep = num_sep__;

	training_set = new int*[N];
	for(int i = 0; i < N; ++i) {
	    training_set[i] = new int[D];
	}

	B = new int*[N];
	for(int i = 0; i < N; ++i) {
	    B[i] = new int[M];
	}
}

void args_for_compute_gradient::SET_epsilon(double* p_epsilon){
	epsilon = *p_epsilon;
}

void args_for_compute_gradient::SET_B(int** B__){
	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < M; ++j){
			B[i][j] = B__[i][j];
		}
	}
}

void args_for_compute_gradient::SET_training_set(int** training_set__){
	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < D; ++j){
			training_set[i][j] = training_set__[i][j];
		}
	}
}

void args_for_compute_gradient::SET_parameters(map<string, double> parameters__){
	parameters["D"] = parameters__["D"];
	parameters["N"] = parameters__["N"];

	parameters["M"] = parameters__["M"];
	parameters["K"] = parameters__["K"];
	parameters["mu"] = parameters__["mu"];
	parameters["init_iter"] = parameters__["init_iter"];
	parameters["init_circle"] = parameters__["init_circle"];
	parameters["opt_iter"] = parameters__["opt_iter"];
	parameters["opt_circle"] = parameters__["opt_circle"];
}

