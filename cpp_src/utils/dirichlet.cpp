#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>
#include <iostream>
#include <math.h>
#include "./dirichlet.h"


std::vector<double> dirichlet_dist(double alpha, int K){
    
    gsl_rng *gen = gsl_rng_alloc (gsl_rng_mt19937);  

    gsl_rng_set(gen,(unsigned)rand());   

    double alpha_arr[K];

    double uniform_dist = 1.0 / (double) K;

    double theta[K];

    for(int i = 0; i < K; i++){
        alpha_arr[i] = alpha;
        theta[i] = uniform_dist;
    }

    gsl_ran_dirichlet(gen, 4, alpha_arr, theta);

    std::vector<double> result(K);

    for(int i = 0; i < 4; i++){
        result[i] = theta[i];
    }

    gsl_rng_free(gen);

    return result;
}