#pragma once
#include "./nn.h"
#include "../utils/utils.h"
#include <torch/script.h>
#include <string>


namespace nn {
    class Connect4NN: public NN{
        
    torch::jit::script::Module net;

    public:
        NNOut eval_state(Board board) override;

        Connect4NN(std::string model_path);

        at::Tensor state_to_tensor(Board board) override;
        at::Tensor visit_count_to_policy_tensor(
            std::map<game::move_id, int>
        ) override;
    };
}

