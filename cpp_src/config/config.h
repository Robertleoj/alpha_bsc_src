#pragma once

#include "../utils/json.h"
#include <fstream>


class config {
public:
    static nlohmann::json hp;
    static void initialize();
};




