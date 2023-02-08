#pragma once
#include "./nn.h"
#include "../utils/utils.h"
#include <torch/script.h>
#include <string>


namespace nn {
    class Connect4NN: public NN{
        
    torch::jit::script::Module net;

    public:
        std::unique_ptr<NNOut> eval_state(Board board) override;

        Connect4NN(std::string model_path);

        at::Tensor state_to_tensor(Board board) override;
        at::Tensor visit_count_to_policy_tensor(
            std::map<game::move_id, int>
        ) override;
        std::vector<std::unique_ptr<NNOut>> eval_states(std::vector<Board> * boards) override;
        
        std::unique_ptr<NNOut> make_nnout_from_tensors(
            at::Tensor policy_tensor,
            at::Tensor value_tensor
        ) override;
    };
}

