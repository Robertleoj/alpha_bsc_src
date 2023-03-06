
#include "./config.h"
#include <ostream>
#include <string>
#include "../utils/utils.h"
#include "../utils/colors.h"

nlohmann::json config::hp;

nlohmann::json read_cfig(std::string fname){
    std::string fpath = utils::string_format("./config_files/%s.json", fname.c_str());

    std::cout << "reading json file at " << fpath <<std::flush;
    
    std::ifstream f(fpath);
    nlohmann::json out = nlohmann::json::parse(f);
    f.close();
    std::cout << colors::GREEN << " [OK]" << colors::RESET << std::endl;
    return out;
}

void config::initialize(){
    config::hp = read_cfig("hyperparameters");
}