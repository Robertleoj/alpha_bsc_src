#include <iostream>
#include <random>
#include <unistd.h>
#include <dirent.h>
#include <torch/all.h>
#include "./simulation/simulation.h"
#include "./config/config.h"
#include "./global.h"
#include "./utils/utils.h"

bool DEBUG = false;

int main(int argc, char *argv[])
{
    std::cout << "PYTORCH VERSION "
              << TORCH_VERSION_MAJOR << '.' 
              << TORCH_VERSION_MINOR << '.' 
              << TORCH_VERSION_PATCH << std::endl;


    // first argument is the run name, second arg is the game name
    if (argc < 2) {
        std::cout << "Usage: ./self_play <run_name> [<game_name>]" << std::endl;
        return 0;
    }

    // get run name
    std::string run_name = argv[1];
    std::string game_name = "connect4";

    // get game name
    if(argc >= 3) {
        game_name = argv[2];
    }

    if (argc == 4) {
        DEBUG = true;
    }

    std::string run_path = "../vault/" + game_name + '/' + run_name;

    // make sure run exists 
    if(!utils::dir_exists(run_path)) {
        std::cout << "Run " << run_name << " does not exist." << std::endl;
        return 1;
    }

    chdir(run_path.c_str());

    // seed random
    srand(time(NULL));

    // initialize config
    config::initialize();

    sim::self_play(game_name);

    return 0;
}
