#include <map>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "util/array_io_txt.h"
#include "algorithm/train.h"

using namespace std;

int main(){

	string filename = "dataset/learn.txt";
	int num_dimension = 128;
	int num_sample = 100000;

	int** training_set = read_from_txt(filename.c_str(), num_sample, num_dimension);

	map<string, double> parameters;
	parameters["N"] = num_sample;
	parameters["D"] = num_dimension;

	parameters["M"] = 2;
	parameters["K"] = 256;
	parameters["mu"] = 0.0001;
	parameters["init_iter"] = 0;
	parameters["init_circle"] = 0;
	parameters["opt_iter"] = 15;
	parameters["opt_circle"] = 10;
	parameters["num_sep"] = 20;

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
				C[i][j][k] = 12 * ((double)rand() / (RAND_MAX));
			}
		}
	}

	int count_B[(int)parameters["M"]];
	for(int i = 0; i < parameters["M"]; ++i){
		count_B[i] = rand() % 10000;
	}

	int** B = new int*[(int)parameters["N"]];
	for(int i = 0; i < parameters["N"]; ++i) {
	    B[i] = new int[(int)parameters["M"]];
	}
	for(int i = 0; i < parameters["N"]; ++i){
		for(int j = 0; j < parameters["M"]; ++j){
			B[i][j] = count_B[j] % (int)parameters["K"];
			// B[i][j] = rand() % (int)parameters["K"];
			count_B[j] = count_B[j] + 1;
		}
	}

	double epsilon = 0;
	double* p_epsilon;
	p_epsilon = &epsilon;

	train(training_set, parameters, C, B, p_epsilon);

	// save optimization results
	ofstream C_txt;
	C_txt.open ("results/C.txt");
	for(int i = 0; i < parameters["M"]; ++i){
		for(int j = 0; j < parameters["D"]; ++j){
			for(int k = 0; k < parameters["K"]; ++k){
				C_txt << C[i][j][k] << " ";
			}
		}
	}
	C_txt.close();

	ofstream B_txt;
	B_txt.open ("results/B.txt");
	for(int i = 0; i < parameters["N"]; ++i){
		for(int j = 0; j < parameters["M"]; ++j){
			B_txt << B[i][j] << " ";
		}
	}
	B_txt.close();

	ofstream epsilon_txt;
	epsilon_txt.open ("results/epsilon.txt");
	epsilon_txt << epsilon;
	epsilon_txt.close();

	cout << "Optimization results saved." << endl;

	for (int i = 0; i < parameters["N"] ; ++i){
    	delete [] training_set[i];
	}
	delete [] training_set;

	for(int i = 0; i < parameters["M"]; ++i) {
		for(int j = 0; j < parameters["D"]; ++j) {
			delete [] C[i][j];
		}
		delete [] C[i];
	}
	delete [] C;

	for (int i = 0; i < parameters["N"] ; ++i){
    	delete [] B[i];
	}
	delete [] B;

	return 0;
}






