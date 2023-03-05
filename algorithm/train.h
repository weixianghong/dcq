#include <map>
#include <cmath>
#include <string>
#include <iostream>

#include "/usr/local/include/lbfgs.h"
#include "opencv2/highgui/highgui.hpp"
#include "../util/array_io_txt.h"
#include "args_for_compute_gradient.h"

using namespace std;
using namespace cv;

void train(int** training_set, map<string, double> parameters, double*** C, int** B, double* p_epsilon);

void binarize(int** B, float** binary_B, int N, int M, int K);

void build_lookup_for_C(map<string, double> parameters, double*** C, double**** lookup_for_C);

double compute_loss(int** training_set, map<string, double> parameters, double*** C, int** B, double* p_epsilon);

void compute_C_grad(int** training_set, map<string, double> parameters, double*** C, int** B, double* p_epsilon, double*** C_grad);

void init_update_B(int** training_set, map<string, double> parameters, double*** C, int** B);

void init_update_C(int** training_set, map<string, double> parameters, double*** C, int** B);

void opt_update_B(int** training_set, map<string, double> parameters, double*** C, int** B, double* p_epsilon);

void opt_update_epsilon(int** training_set, map<string, double> parameters, double*** C, int** B, double* p_epsilon);

void opt_update_C(int** training_set, map<string, double> parameters, double*** C, int** B, double* p_epsilon);
