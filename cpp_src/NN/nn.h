#pragma once

#include "../base/board.h"
#include "../games/game.h"
#include "../base/types.h"
#include <map>
#include <memory>
#include <torch/torch.h>
#include <vector>
#include <string>


namespace nn {


    
    typedef std::map<game::move_id, double> move_dist;

    struct NNOut {
        move_dist p;
        double v;
    };

    class NN {
    public:
        virtual std::unique_ptr<NNOut> eval_state(Board board) = 0;

        virtual std::vector<std::unique_ptr<NNOut>> eval_states(std::vector<Board> * boards) = 0;

        virtual std::vector<std::unique_ptr<NNOut>> eval_tensors(std::vector<at::Tensor>&) = 0;
        virtual std::vector<std::unique_ptr<NNOut>> eval_batch(at::Tensor) = 0;

        virtual at::Tensor state_to_tensor(Board board) = 0;
        virtual at::Tensor prepare_batch(std::vector<at::Tensor>&) = 0;

        virtual at::Tensor move_map_to_policy_tensor(move_dist) = 0;
        virtual std::unique_ptr<NNOut> make_nnout_from_tensors(at::Tensor policy, at::Tensor value) = 0;
    };
}