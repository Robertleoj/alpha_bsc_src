#pragma once
#include "./nn.h"
#include "../utils/utils.h"
#include <torch/script.h>
#include <string>


namespace nn {
    class BreakthroughNN: public NN{

    public:
        BreakthroughNN(std::string model_path);

        at::Tensor pol_softmax(at::Tensor) override;
        at::Tensor state_to_tensor(Board board) override;
        at::Tensor move_map_to_policy_tensor(
            move_dist,
            pp::Player
        ) override;
        
        std::unique_ptr<NNOut> make_nnout_from_tensors(
            at::Tensor policy_tensor,
            at::Tensor value_tensor
        ) override;

        std::unique_ptr<NNOut> make_nnout_from_tensors(
            at::Tensor policy_tensor,
            at::Tensor value_tensor,
            std::vector<game::move_id> *,
            pp::Player player
        ) override;
    };
}

