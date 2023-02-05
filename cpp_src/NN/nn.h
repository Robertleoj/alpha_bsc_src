#pragma once

#include "../base/board.h"
#include "../games/game.h"
#include <map>
#include <memory>
#include <torch/torch.h>


namespace nn {

    struct TrainingSample {
        at::Tensor target_policy;
        at::Tensor state;
        double outcome;
    };
    
    struct NNOut {
        std::map<game::move_id, double> p;
        double v;
    };

    class NN {
    public:
        virtual NNOut eval_state(Board board){};
        virtual at::Tensor state_to_tensor(Board board){}
        virtual at::Tensor visit_count_to_policy_tensor(
            std::map<game::move_id, int>
        ){}
    };
}