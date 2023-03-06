#include <iostream>
#include <random>
#include <torch/all.h>
#include "games/connect4.h"
#include "./simulation/simulation.h"
#include "./config/config.h"


int main()
{
    std::cout << "PYTORCH VERSION "
              << TORCH_VERSION_MAJOR << '.' 
              << TORCH_VERSION_MINOR << '.' 
              << TORCH_VERSION_PATCH << std::endl;


    // seed random
    srand(time(NULL));

    // initialize config
    config::initialize();

    sim::eval_targets("../db/test_data.json", -1);

    return 0;
}
