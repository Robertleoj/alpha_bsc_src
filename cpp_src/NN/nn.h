#pragma once

#include "../base/board.h"
#include "../games/game.h"
#include <map>
#include <memory>

namespace nn {
    
    struct NNOut {
        std::map<game::move_id, double> p;
        double v;
    };

    class NN {
    public:
        virtual NNOut eval_state(Board board){};
    };
}