
#include "./config.h"
#include <string>
#include "../utils/utils.h"

nlohmann::json config::hp;
nlohmann::json config::db;

nlohmann::json read_cfig(std::string fname){
    std::string fpath = utils::string_format("./config_files/%s.json", fname.c_str());

    std::cout << "reading json file at " << fpath << std::endl;
    
    std::ifstream f(fpath);
    nlohmann::json out = nlohmann::json::parse(f);
    f.close();
    return out;
}

void config::initialize(){
    config::hp = read_cfig("hyperparameters");
    config::db = read_cfig("db");
}