#include <iostream>
#include <random>
#include <torch/all.h>
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

    sim::self_play("connect4");

    return 0;
}
