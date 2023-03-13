#include <dirent.h>
#include "./utils.h"

namespace utils{

    bool dir_exists(const std::string& path) {
        DIR* dir = opendir(path.c_str());
        if(!dir){
            return false;
        }
        closedir(dir);
        return true;
    }
}