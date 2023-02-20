#pragma once

#include "../base/board.h"
#include "../games/game.h"
#include <map>
#include <memory>
#include <torch/torch.h>
#include <vector>


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
        virtual std::unique_ptr<NNOut> eval_state(Board board){};
        virtual std::vector<std::unique_ptr<NNOut>> eval_states(std::vector<Board> * boards){};

        virtual std::vector<std::unique_ptr<NNOut>> eval_tensors(std::vector<at::Tensor>&) {};
        virtual at::Tensor state_to_tensor(Board board){}
        virtual at::Tensor visit_count_to_policy_tensor(
            std::map<game::move_id, int>
        ){}
        virtual std::unique_ptr<NNOut> make_nnout_from_tensors(at::Tensor policy, at::Tensor value){}
    };
}