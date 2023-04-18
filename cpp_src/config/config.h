#pragma once

#include "../utils/json.h"
#include <fstream>


class config {
public:
    static nlohmann::json hp;
    static void initialize();
    static bool has_key(const std::string& param);
};




