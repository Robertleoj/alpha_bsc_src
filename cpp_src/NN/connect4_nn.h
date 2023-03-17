#pragma once
#include "./nn.h"
#include "../utils/utils.h"
#include <torch/script.h>
#include <string>


namespace nn {
    class Connect4NN: public NN{
        


    public:
        Connect4NN(std::string model_path);

        std::unique_ptr<NNOut> eval_state(Board board) override;

        std::vector<std::unique_ptr<NNOut>> eval_tensors(std::vector<at::Tensor>&) override;


        at::Tensor prepare_batch(std::vector<at::Tensor>&) override;
        std::vector<std::unique_ptr<NNOut>> eval_batch(at::Tensor) override;
        at::Tensor state_to_tensor(Board board) override;
        at::Tensor move_map_to_policy_tensor(move_dist) override;
        std::vector<std::unique_ptr<NNOut>> eval_states(std::vector<Board> * boards) override;
        
        std::unique_ptr<NNOut> make_nnout_from_tensors(
            at::Tensor policy_tensor,
            at::Tensor value_tensor
        ) override;

    private:
        torch::jit::script::Module net;

    };
}

