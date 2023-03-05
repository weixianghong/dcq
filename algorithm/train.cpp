#include "train.h"

static lbfgsfloatval_t evaluate(void *instance,
							    const lbfgsfloatval_t *x,
							    lbfgsfloatval_t *g,
							    const int n,
							    const lbfgsfloatval_t step
							    ) {
	args_for_compute_gradient* args = static_cast<args_for_compute_gradient*>(instance);

	double epsilon = args->epsilon;
	int** B = args->B;
	int** training_set = args->training_set;

	map<string, double> parameters;
	parameters["D"] = args->parameters["D"];
	parameters["N"] = args->parameters["N"];

	parameters["M"] = args->parameters["M"];
	parameters["K"] = args->parameters["K"];
	parameters["mu"] = args->parameters["mu"];
	parameters["init_iter"] = args->parameters["init_iter"];
	parameters["init_circle"] = args->parameters["init_circle"];
	parameters["opt_iter"] = args->parameters["opt_iter"];
	parameters["opt_circle"] = args->parameters["opt_circle"];
	parameters["num_sep"] = args->parameters["num_sep"];

	int M = parameters["M"];
	int K = parameters["K"];

	int N = parameters["N"];
	int D = parameters["D"];

	double*** C = new double**[M];
	for(int i = 0; i < M; ++i) {
	    C[i] = new double*[D];
		for(int j = 0; j < D; ++j) {
			C[i][j] = new double[K];
		}
	}
	#pragma omp parallel for
	for(int i = 0; i < M; ++i){
		for(int j = 0; j < D; ++j){
			for(int k = 0; k < K; ++k){
				// C[i][j][k] = rand() % 256;
				C[i][j][k] = x[i * D * K + j * K + k];
			}
		}
	}

	double*** C_grad = new double**[M];
	for(int i = 0; i < M; ++i) {
	    C_grad[i] = new double*[D];
		for(int j = 0; j < D; ++j) {
			C_grad[i][j] = new double[K];
		}
	}

	compute_C_grad(training_set, parameters, C, B, &epsilon, C_grad);

	#pragma omp parallel for
	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < D; ++j) {
			for(int k = 0; k < K; ++k) {
 				g[i * D * K + j * K + k] = C_grad[i][j][k];
			}
		}
	}

	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < D; ++j) {
			delete [] C_grad[i][j];
		}
		delete [] C_grad[i];
	}
	delete [] C_grad;

	double total_loss = compute_loss(training_set, parameters, C, B, &epsilon);
	return total_loss;
}

static int progress(void *instance,
				    const lbfgsfloatval_t *x,
				    const lbfgsfloatval_t *g,
				    const lbfgsfloatval_t fx,
				    const lbfgsfloatval_t xnorm,
				    const lbfgsfloatval_t gnorm,
				    const lbfgsfloatval_t step,
				    int n,
				    int k,
				    int ls
				    ) {
    // printf("Iteration %d:\n", k);
    printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
    // printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    // printf("\n");
    return 0;
}

void train(int** training_set, map<string, double> parameters, double*** C, int** B, double* p_epsilon){

	int M = parameters["M"];
	int K = parameters["K"];

	int N = parameters["N"];
	int D = parameters["D"];

	cout.precision(20);

	double loss = compute_loss(training_set, parameters, C, B, p_epsilon);
	cout << "Initial loss: " << loss << endl;

	// initilize
	int init_iter = parameters["init_iter"];
	for(int iteration = 0; iteration < init_iter; ++iteration) {

		cout << "Init iteration " << iteration << ":" << endl;

		init_update_B(training_set, parameters, C, B);
		loss = compute_loss(training_set, parameters, C, B, p_epsilon);
		cout << "    Loss after updating B: " << loss << endl;

		init_update_C(training_set, parameters, C, B);
		loss = compute_loss(training_set, parameters, C, B, p_epsilon);
		cout << "    Loss after updating C: " << loss << endl << endl;
	}

	// optimize
	int** count_selection = new int*[M];
	for(int i = 0; i < M; ++i) {
	    count_selection[i] = new int[K];
	}

	int opt_iter = parameters["opt_iter"];
	for(int iteration = 0; iteration < opt_iter; ++iteration) {

		cout << "Opt iteration " << iteration << ":" << endl;

		// for any word in C that is not selected as component of any data, reinitialize it
		int count_idle_word = 0;
		for(int i = 0; i < M; ++i){
			for(int j = 0; j < K; ++j){
				count_selection[i][j] = 0;
			}
		}
		for(int i = 0; i < N; ++i){
			for(int j = 0; j < M; ++j){
				count_selection[j][ B[i][j] ] = count_selection[j][ B[i][j] ] + 1;
			}
		}
		for(int i = 0; i < parameters["M"]; ++i){
			for(int j = 0; j < parameters["D"]; ++j){
				for(int k = 0; k < parameters["K"]; ++k){
					if( count_selection[i][k] == 0){
						count_idle_word = count_idle_word + 1;
						C[i][j][k] = 5 * ((double)rand() / (RAND_MAX));
					}
				}
			}
		}
		cout << "    " << count_idle_word/D << " idle words are found." << endl;
		/////////////////////////////////////////////////////////////////////////////////////

		opt_update_B(training_set, parameters, C, B, p_epsilon);
		loss = compute_loss(training_set, parameters, C, B, p_epsilon);
		cout << "    Loss after updating B: " << loss << endl;

		opt_update_epsilon(training_set, parameters, C, B, p_epsilon);
		loss = compute_loss(training_set, parameters, C, B, p_epsilon);
		cout << "    Loss after updating e: " << loss << endl;

		opt_update_C(training_set, parameters, C, B, p_epsilon);
		loss = compute_loss(training_set, parameters, C, B, p_epsilon);
		cout << "    Loss after updating C: " << loss << endl << endl;
	}
	for (int i = 0; i < parameters["M"] ; ++i){
    	delete [] count_selection[i];
	}
	delete [] count_selection;

	loss = compute_loss(training_set, parameters, C, B, p_epsilon);
	cout << "Final loss: " << loss << endl;

}

void init_update_B(int** training_set, map<string, double> parameters, double*** C, int** B){

	int M = parameters["M"];
	int K = parameters["K"];

	int N = parameters["N"];
	int D = parameters["D"];

	int init_circle = parameters["init_circle"];
	
	for(int circle = 0; circle < init_circle; ++circle) {
		#pragma omp parallel for
		for(int i = 0; i < N; ++i) {
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
				
				double min_distortion = 0;
				for(int k = 0; k < K; ++k) {
					double approximate[D];
					for(int dim = 0; dim < D; ++dim) {
						approximate[dim] = sum_of_other_component[dim] + C[j][dim][k];
					}

					double residual[D];
					for(int dim = 0; dim < D; ++dim) {
						residual[dim] = training_set[i][dim] - approximate[dim];
					}

					//for(int dim = 0; dim < D; ++dim) {
					//	cout << residual[dim] << " ";
					//}
					//cout << endl;

					double distortion_square = 0.;
				    for(int dim = 0; dim < D; ++dim) {
				        distortion_square += residual[dim] * residual[dim];
				    }

				    double distortion = distortion_square;

					double min_distortion;
					if(k == 0) {
						min_distortion = distortion;
						B[i][j] = 0;
					}
					if(distortion < min_distortion) {
						min_distortion = distortion;
						B[i][j] = k;
					}
				}
				
			}
		}
	}
	
}

void init_update_C(int** training_set, map<string, double> parameters, double*** C, int** B) {

	int M = parameters["M"];
	int K = parameters["K"];

	int N = parameters["N"];
	int D = parameters["D"];

	float** binary_B = new float*[N];
	for(int i = 0; i < N; ++i) {
	    binary_B[i] = new float[M * K];
	}
	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < M * K; ++j){
			binary_B[i][j] = 0;
		}
	}

	binarize(B, binary_B, N, M, K);	

	Mat Mat_binary_B;
	Mat_binary_B.create(N, M * K, CV_32F);
	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < M * K; ++j){
			Mat_binary_B.at<float>(i, j) = binary_B[i][j];
		}
	}

	Mat Mat_binary_B_in_paper;
	transpose(Mat_binary_B, Mat_binary_B_in_paper);

	Mat Mat_BB_T = Mat_binary_B_in_paper * Mat_binary_B;
	Mat Mat_BB_T_inv;
	invert(Mat_BB_T, Mat_BB_T_inv, DECOMP_SVD);

	// float** float_training_set = new float*[N];
	// for(int i = 0; i < N; ++i) {
	//    float_training_set[i] = new float[D];
	// }

	Mat Mat_float_training_set;
	Mat_float_training_set.create(N, D, CV_32F);
	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < D; ++j){
			Mat_float_training_set.at<float>(i, j) = training_set[i][j];
		}
	}

	Mat Mat_float_training_set_in_paper;
	transpose(Mat_float_training_set, Mat_float_training_set_in_paper);

	Mat Mat_C_2d_array = Mat_float_training_set_in_paper * Mat_binary_B * Mat_BB_T_inv;

	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < D; ++j){
			for(int k = 0; k < K; ++k){
				C[i][j][k] = Mat_C_2d_array.at<float>(j, i*K + k);
			}
		}
	}

	for (int i = 0; i < N ; ++i){
    	delete [] binary_B[i];
	}
	delete [] binary_B;

	// for (int i = 0; i < N ; ++i){
    //	delete [] float_training_set[i];
	// }
	// delete [] float_training_set;

}

void binarize(int** B, float** binary_B, int N, int M, int K){

	int index = 0;
	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < M; ++j) {
			index = j * K + B[i][j];
			binary_B[i][index] = 1;
		}
	}

}

void opt_update_B(int** training_set, map<string, double> parameters, double*** C, int** B, double* p_epsilon){

	int M = parameters["M"];
	int K = parameters["K"];

	int N = parameters["N"];
	int D = parameters["D"];

	double mu = parameters["mu"];

	int opt_circle = parameters["opt_circle"];
	
	for(int circle = 0; circle < opt_circle; ++circle) {
		#pragma omp parallel for
		for(int i = 0; i < N; ++i) {
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
						residual[dim] = training_set[i][dim] - approximate[dim];
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
					constraint_loss = constraint_loss - *p_epsilon;
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
}


void build_lookup_for_C(map<string, double> parameters, double*** C, double**** lookup_for_C) {

	int M = parameters["M"];
	int K = parameters["K"];
	int D = parameters["D"];

	#pragma omp parallel for
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


void opt_update_epsilon(int** training_set, map<string, double> parameters, double*** C, int** B, double* p_epsilon){

	int M = parameters["M"];
	int K = parameters["K"];

	int N = parameters["N"];
	int D = parameters["D"];

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

	double accumulated_inner_product = 0;
	double* accumulated_inner_product_vec = new double[N];

	#pragma omp parallel for
	for(int i = 0; i < N; ++i) {
		double inner_product = 0;
		for(int pp = 0; pp < M; ++pp) {
			for(int qq = 0; qq < M; ++qq) {
				if(pp != qq) {
					inner_product += lookup_for_C[pp] [B[i][pp]] [qq] [B[i][qq]];
				}
			}
		}
		accumulated_inner_product_vec[i] = inner_product;
	}

	for(int i = 0; i < N; ++i) {	
		accumulated_inner_product += accumulated_inner_product_vec[i];
	}

	*p_epsilon = accumulated_inner_product / N;

	delete [] accumulated_inner_product_vec;

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

double compute_loss(int** training_set, map<string, double> parameters, double*** C, int** B, double* p_epsilon){

	int M = parameters["M"];
	int K = parameters["K"];

	int N = parameters["N"];
	int D = parameters["D"];

	double mu = parameters["mu"];

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

	double first_term = 0;
	double second_term = 0;

	double* first_term_vec = new double[N];
	double* second_term_vec = new double[N];

	#pragma omp parallel for
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

		double residual[D];
		for(int dim = 0; dim < D; ++dim) {
			residual[dim] = training_set[i][dim] - approximate[dim];
		}

		double distortion_square = 0.;
	    for(int dim = 0; dim < D; ++dim) {
	        distortion_square += residual[dim] * residual[dim];
	    }

	    first_term_vec[i] = distortion_square;

		double constraint_loss = 0;
		for(int pp = 0; pp < M; ++pp) {
			for(int qq = 0; qq < M; ++qq) {
				if(pp != qq){
					constraint_loss += lookup_for_C[pp] [B[i][pp]] [qq] [B[i][qq]];
				}
			}
		}
		constraint_loss = constraint_loss - *p_epsilon;
		constraint_loss = constraint_loss * constraint_loss;

		second_term_vec[i] = constraint_loss;
	}

	for(int i = 0; i < N; ++i) {
		first_term += first_term_vec[i];
		second_term += second_term_vec[i];
	}

	double total_loss = first_term + mu * second_term;

	delete [] first_term_vec;
	delete [] second_term_vec;

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

	return total_loss;
}

void compute_C_grad(int** training_set, map<string, double> parameters, double*** C, int** B, double* p_epsilon, double*** C_grad) {

	int M = parameters["M"];
	int K = parameters["K"];

	int N = parameters["N"];
	int D = parameters["D"];

	double mu = parameters["mu"];

	int num_sep = parameters["num_sep"];

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


	double**** C_grad_multi_sep = new double***[num_sep];
	for(int sep = 0; sep < num_sep; ++sep) {
		C_grad_multi_sep[sep] = new double**[M];
		for(int i = 0; i < M; ++i) {
		    C_grad_multi_sep[sep][i] = new double*[D];
			for(int j = 0; j < D; ++j) {
				C_grad_multi_sep[sep][i][j] = new double[K];
			}
		}
	}

	for(int sep = 0; sep < num_sep; ++sep) {
		for(int i = 0; i < M; ++i) {
			for(int j = 0; j < D; ++j) {
				for(int k = 0; k < K; ++k) {
					C_grad_multi_sep[sep][i][j][k] = 0.0;
				}
			}
		}
	}


	#pragma omp parallel for
	for (int sep = 0; sep < num_sep; ++sep)
	{
		int start_point_id = N / num_sep * sep;
		int end_point_id = N / num_sep * (sep + 1);

		for(int i = start_point_id; i < end_point_id; ++i) {

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
						C_grad_multi_sep[sep][j][pp][qq] += 2 * negative_residual[pp] * B_im[qq];
						// C_grad[j][pp][qq] += 2 * negative_residual[pp] * B_im[qq];
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
						C_grad_multi_sep[sep][j][pp][qq] += second_term_multiplier * sum_of_other_component[pp] * B_im[qq];
						// C_grad[j][pp][qq] += second_term_multiplier * sum_of_other_component[pp] * B_im[qq];
					}
				}
			}
		}
	}

	// merge
	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < D; ++j) {
			for(int k = 0; k < K; ++k) {
				for(int sep = 0; sep < num_sep; ++sep) {
					C_grad[i][j][k] += C_grad_multi_sep[sep][i][j][k];
				}
			}
		}
	}
	//

	for(int sep = 0; sep < num_sep; ++sep) {
		for(int i = 0; i < M; ++i) {
			for(int j = 0; j < D; ++j) {
				delete [] C_grad_multi_sep[sep][i][j];
			}
			delete [] C_grad_multi_sep[sep][i];
		}
		delete [] C_grad_multi_sep[sep];
	}
	delete [] C_grad_multi_sep;

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

void opt_update_C(int** training_set, map<string, double> parameters, double*** C, int** B, double* p_epsilon){

	int M = parameters["M"];
	int D = parameters["D"];
	int K = parameters["K"];
	int N = parameters["N"];

	args_for_compute_gradient args(N,M,D,K);
	args.SET_epsilon(p_epsilon);
	args.SET_B(B);
	args.SET_training_set(training_set);
	args.SET_parameters(parameters);

	lbfgsfloatval_t function_value;
	lbfgsfloatval_t* x = lbfgs_malloc(M * D * K);

	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < D; ++j) {
			for(int k = 0; k < K; ++k) {
 				x[i * D * K + j * K + k] = C[i][j][k];
			}
		}
	}

    lbfgs_parameter_t lbfgs_param;
    lbfgs_parameter_init(&lbfgs_param);
	lbfgs(M*D*K, x, &function_value, evaluate, NULL, &args, &lbfgs_param);

	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < D; ++j) {
			for(int k = 0; k < K; ++k) {
 				C[i][j][k] = x[i * D * K + j * K + k];
			}
		}
	}
	lbfgs_free(x);
}


