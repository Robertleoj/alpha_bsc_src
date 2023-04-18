#include "./config.h"
#include <ostream>
#include <string>
#include "../utils/utils.h"
#include "../utils/colors.h"

nlohmann::json config::hp;

nlohmann::json read_cfig(){
    std::string fpath = "./cpp_hyperparameters.json";

    std::cout << "reading json file at " << fpath <<std::flush;
    
    std::ifstream f(fpath);
    nlohmann::json out = nlohmann::json::parse(f);
    f.close();
    std::cout << colors::GREEN << " [OK]" << colors::RESET << std::endl;
    return out;
}

void config::initialize(){
    config::hp = read_cfig();
}

bool config::has_key(const std::string& key){
    return config::hp.find(key) != config::hp.end();
}