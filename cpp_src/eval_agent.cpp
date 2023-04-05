#include <iostream>
#include <random>
#include <torch/all.h>
#include <unistd.h>
#include "games/connect4.h"
#include "./simulation/simulation.h"
#include "./utils/utils.h"
#include "./config/config.h"
#include "./global.h"

const bool DEBUG = false;

int main(int argc, char *argv[])
{

    if(argc < 2){
        std::cout << "Usage: ./eval_agent <run_name>" << std::endl;
        return 1;
    }

    std::string run_name = argv[1];

    std::string game_name = "connect4";

    std::string run_path = "../vault/" + game_name + '/' + run_name;

    int generation = -1;
    if(argc == 3){
        generation = atoi(argv[2]);
    }

    if(!utils::dir_exists(run_path)) {
        std::cout << "Run " << run_name << " does not exist." << std::endl;
        return 1;
    }

    chdir(run_path.c_str());


    std::cout << "PYTORCH VERSION "
              << TORCH_VERSION_MAJOR << '.' 
              << TORCH_VERSION_MINOR << '.' 
              << TORCH_VERSION_PATCH << std::endl;


    // seed random
    srand(time(NULL));


    // initialize config
    config::initialize();

    sim::eval_targets("../../../db/test_data.json", generation);

    return 0;
}
