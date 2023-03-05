#include <map>
#include <string>
#include <stdlib.h>
#include <iostream>

#include "../../util/array_io_txt.h"

using namespace std;

void build_lookup_for_C(map<string, double> parameters, double*** C, double**** lookup_for_C) {

	int M = parameters["M"];
	int K = parameters["K"];
	int D = parameters["D"];

	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < K; ++j) {
			for(int p = 0; p < M; ++p) {
				for(int q = 0; q < K; ++q) {
					lookup_for_C[i][j][p][q] = 0;
				}
			}	
		}
	}	

	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < K; ++j) {
			for(int p = 0; p < M; ++p) {
				for(int q = 0; q < K; ++q) {
					for(int dim = 0; dim < D; ++dim) {
				        lookup_for_C[i][j][p][q] += C[i][dim][j] * C[p][dim][q];
				    }
				}
			}	
		}
	}	

}

void compute_C_grad(int** training_set, map<string, double> parameters, double*** C, int** B, double* p_epsilon, double*** C_grad) {

	int M = parameters["M"];
	int K = parameters["K"];

	int N = parameters["N"];
	int D = parameters["D"];

	double mu = parameters["mu"];

	for(int i = 0; i < M; ++i){
		for(int j = 0; j < D; ++j){
			for(int k = 0; k < K; ++k){
				C_grad[i][j][k] = 0.0;
			}
		}
	}

	double**** lookup_for_C = new double***[M];
	for(int i = 0; i < M; ++i) {
	    lookup_for_C[i] = new double**[K];
		for(int j = 0; j < K; ++j) {
			lookup_for_C[i][j] = new double*[M];
			for(int p = 0; p < M; ++p) {
				lookup_for_C[i][j][p] = new double[K];
			}
		}
	}

	build_lookup_for_C(parameters, C, lookup_for_C);

	for(int i = 0; i < N; ++i) {

		double approximate[D];
		for(int dim = 0; dim < D; ++dim) {
			approximate[dim] = 0;
		}
		for(int j = 0; j < M; ++j) {
			for(int dim = 0; dim < D; ++dim) {
				approximate[dim] += C[j] [dim] [B[i][j]];
			}
		}

		double negative_residual[D];
		for(int dim = 0; dim < D; ++dim) {
			negative_residual[dim] = approximate[dim] - training_set[i][dim];
		}

		for(int j = 0; j < M; ++j) {
			int B_im[K];
			for(int jj = 0; jj < K; ++jj) {
				B_im[jj] = 0;
			}
			B_im[B[i][j]] = 1;

			for(int pp = 0; pp < D; ++pp) {
				for(int qq = 0; qq < K; ++qq) {
					C_grad[j][pp][qq] += 2 * negative_residual[pp] * B_im[qq];
				}
			}
		}

		//////////////////////
		double second_term_multiplier = 0;
		for(int pp = 0; pp < M; ++pp) {
			for(int qq = 0; qq < M; ++qq) {
				if(pp != qq){
					second_term_multiplier += lookup_for_C[pp] [B[i][pp]] [qq] [B[i][qq]];
				}
			}
		}
		second_term_multiplier = second_term_multiplier - *p_epsilon;
		second_term_multiplier = 4 * mu * second_term_multiplier;

		for(int j = 0; j < M; ++j) {
			double sum_of_other_component[D];
			for(int dim = 0; dim < D; ++dim) {
				sum_of_other_component[dim] = 0;
			}
			for(int other_dict = 0; other_dict < M; ++other_dict) {
				if(other_dict != j) {
					for(int dim = 0; dim < D; ++dim) {
						sum_of_other_component[dim] += C[other_dict] [dim] [B[i][other_dict]];
					}
				}
			}

			int B_im[K];
			for(int jj = 0; jj < K; ++jj) {
				B_im[jj] = 0;
			}
			B_im[B[i][j]] = 1;

			for(int pp = 0; pp < D; ++pp) {
				for(int qq = 0; qq < K; ++qq) {
					C_grad[j][pp][qq] += second_term_multiplier * sum_of_other_component[pp] * B_im[qq];
				}
			}
		}
	}

	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < K; ++j) {
			for(int p = 0; p < M; ++p) {
				delete [] lookup_for_C[i][j][p];
			}
			delete [] lookup_for_C[i][j];
		}
		delete [] lookup_for_C[i];
	}
	delete [] lookup_for_C;

}


int main(int argc, char *argv[])
{

	cout.precision(15);

	string filename = "part.txt";
	int** training_set = read_from_txt(filename.c_str());

	map<string, double> parameters;
	parameters["D"] = 128;
	parameters["N"] = 500;

	parameters["M"] = 2;
	parameters["K"] = 1;
	parameters["mu"] = 0.0004;
	parameters["init_iter"] = 1;
	parameters["init_circle"] = 1;
	parameters["opt_iter"] = 1;
	parameters["opt_circle"] = 3;
	parameters["multi_process"] = 20;

	int M = parameters["M"];
	int D = parameters["D"];
	int K = parameters["K"];
	int N = parameters["N"];

	double*** C = new double**[(int)parameters["M"]];
	for(int i = 0; i < parameters["M"]; ++i) {
	    C[i] = new double*[(int)parameters["D"]];
		for(int j = 0; j < parameters["D"]; ++j) {
			C[i][j] = new double[(int)parameters["K"]];
		}
	}
	for(int i = 0; i < parameters["M"]; ++i){
		for(int j = 0; j < parameters["D"]; ++j){
			for(int k = 0; k < parameters["K"]; ++k){
				// C[i][j][k] = rand() % 256;
				C[i][j][k] = 0.3*i + 0.9*j + 0.2*k;
			}
		}
	}

	int count_B = 0;

	int** B = new int*[(int)parameters["N"]];
	for(int i = 0; i < parameters["N"]; ++i) {
	    B[i] = new int[(int)parameters["M"]];
	}
	for(int i = 0; i < parameters["N"]; ++i){
		for(int j = 0; j < parameters["M"]; ++j){
			B[i][j] = count_B % K;
			count_B++;
		}
	}

	double epsilon = 0;


	double*** C_grad = new double**[M];
	for(int i = 0; i < M; ++i) {
	    C_grad[i] = new double*[D];
		for(int j = 0; j < D; ++j) {
			C_grad[i][j] = new double[K];
		}
	}

	compute_C_grad(training_set, parameters, C, B, &epsilon, C_grad);

	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < D; ++j) {
			delete [] C_grad[i][j];
		}
		delete [] C_grad[i];
	}
	delete [] C_grad;

	for (int i = 0; i < parameters["N"] ; ++i){
    	delete [] training_set[i];
	}
	delete [] training_set;

	for (int i = 0; i < parameters["N"] ; ++i){
    	delete [] B[i];
	}
	delete [] B;

}





