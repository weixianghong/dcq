#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>

using namespace std;

int main(){

	int	num_sample = 1000000;
	int	num_dimension = 128;

	int** base = new int*[num_sample];
	for(int i = 0; i < num_sample; ++i) {
    	base[i] = new int[num_dimension];
	}

	ifstream infile("sift1m_base.txt");
	string line;
	int i = 0, j = 0, k = 0;
	
	while (getline(infile, line)) {
		istringstream iss(line);
		int temp;

		j = 0;
		while (iss >> temp) {
			base[i][j] = temp;
			j++;
		}
	i++;
	}

	infile.close();





	map<string, double> parameters;
	parameters["N"] = num_sample;
	parameters["D"] = num_dimension;

	parameters["M"] = 2;
	parameters["K"] = 256;
	parameters["mu"] = 0.0001;
	parameters["init_iter"] = 3;
	parameters["init_circle"] = 3;
	parameters["opt_iter"] = 15;
	parameters["opt_circle"] = 5;
	parameters["num_sep"] = 20;

	double*** C = new double**[(int)parameters["M"]];
	for(int i = 0; i < parameters["M"]; ++i) {
	    C[i] = new double*[(int)parameters["D"]];
		for(int j = 0; j < parameters["D"]; ++j) {
			C[i][j] = new double[(int)parameters["K"]];
		}
	}

	ifstream C_file("../results/C.txt");
	int count = 0, rest = 0;
	
	getline(C_file, line);
	istringstream iss(line);
	double temp;

	while (iss >> temp) {
				
		i = count / (int)(parameters["D"] * parameters["K"]);
		rest = count - i * (int)(parameters["D"] * parameters["K"]);
		j = rest / (int)(parameters["K"]);
		k = rest - j * (int)(parameters["K"]);

		C[i][j][k] = temp;
		count++;
	}

	C_file.close();


	int count_B = 0;
	int** B = new int*[(int)parameters["N"]];
	for(int i = 0; i < parameters["N"]; ++i) {
	    B[i] = new int[(int)parameters["M"]];
	}
	for(int i = 0; i < parameters["N"]; ++i){
		for(int j = 0; j < parameters["M"]; ++j){
			B[i][j] = count_B % (int)parameters["K"];
			// B[i][j] = rand() % (int)parameters["K"];
			count_B++;
		}
	}


	double epsilon;
	ifstream epsilon_file("../results/epsilon.txt");
	
	getline(epsilon_file, line);
	istringstream iss_3(line);
	iss_3 >> epsilon;

	epsilon_file.close();
	cout << epsilon << endl;








	int M = parameters["M"];
	int K = parameters["K"];

	int N = parameters["N"];
	int D = parameters["D"];

	double mu = parameters["mu"];

	int opt_circle = parameters["opt_circle"];
	
	for(int circle = 0; circle < opt_circle; ++circle) {
		#pragma omp parallel for
		for(int i = 0; i < N; ++i) {
			cout << i << endl;
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
				
				double min_summation = 0;
				int min_summation_k = 0;

				double min_distortion = 0;

				for(int k = 0; k < K; ++k) {

					// first term
					double approximate[D];
					for(int dim = 0; dim < D; ++dim) {
						approximate[dim] = sum_of_other_component[dim] + C[j][dim][k];
					}

					double residual[D];
					for(int dim = 0; dim < D; ++dim) {
						residual[dim] = base[i][dim] - approximate[dim];
					}

					double distortion_square = 0.;
				    for(int dim = 0; dim < D; ++dim) {
				        distortion_square += residual[dim] * residual[dim];
				    }

				    double distortion = distortion_square;

					// second term
					int B_i[M];
					for(int jj = 0; jj < M; ++jj) {
						B_i[jj] = B[i][jj];
					}
					B_i[j] = k;

					double constraint_loss = 0;
					for(int ii = 0; ii < M; ++ii) {
						for(int jj = 0; jj < M; ++jj) {
							if(jj != ii) {
								for(int dim = 0; dim < D; ++dim) {
									constraint_loss += C[ii][dim][B_i[ii]] * C[jj][dim][B_i[jj]];
								}
							}
						}
					}
					constraint_loss = constraint_loss - epsilon;
					constraint_loss = constraint_loss * constraint_loss;
					constraint_loss = mu * constraint_loss;

					// choose the k that minimizes summation
					double summation = distortion + constraint_loss;
					if(k == 0) {
						min_summation = summation;
					}
					if(summation < min_summation) {
						min_summation = summation;
						min_summation_k = k;
					}
				}
				B[i][j] = min_summation_k;
			}
		}
	}


	ofstream B_for_base;
	B_for_base.open ("B_for_base.txt");
	for(int i = 0; i < parameters["N"]; ++i){
		for(int j = 0; j < parameters["M"]; ++j){
			B_txt << B[i][j] << " ";
		}
	}
	B_for_base.close();


	return 0;
}











