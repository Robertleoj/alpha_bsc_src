#include <string>
#include "./breakthrough_nn.h"
#include "../base/types.h"
#include "../utils/utils.h"
#include <fstream>
#include <cuda_runtime.h>



namespace nn{
    
    /**
     * @brief Constructor
     * 
     * @param model_path 
     */
    BreakthroughNN::BreakthroughNN(std::string model_path): NN(model_path){}

    at::Tensor BreakthroughNN::pol_softmax(at::Tensor pol_tensor){
        return pol_tensor
                .reshape({-1, 64 * 64})
                .softmax(1)
                .reshape({-1, 64, 64});
    }

    /**
     * @brief Create a NNOut from the policy and value tensors
     * 
     * @param pol_tensor 
     * @param val_tensor 
     * @return std::unique_ptr<NNOut> 
     */
    std::unique_ptr<NNOut> BreakthroughNN::make_nnout_from_tensors(at::Tensor pol_tensor, at::Tensor val_tensor){

        float * flat = pol_tensor.flatten().contiguous().data_ptr<float>();

        
        nn::move_dist p;
        p.reserve(64 * 64);

        // std::array<std::pair<game::move_id, double>, 64 * 64> init_list;

        for(int i = 0; i < 64 * 64; i++){
            p.emplace(((i / 64) << 6) | (i % 64), (double) flat[i]);
            // init_list[i] = std::make_pair(((i / 64) << 6) | (i % 64), (double) flat[i]);
        }

        // p.insert(init_list.begin(), init_list.end());

        return std::unique_ptr<NNOut>(new NNOut{
            std::move(p),
            val_tensor.item().toDouble()
        });
    }
    
    /**
     * @brief Convert a state to a tensor
     * 
     * @param board 
     * @return at::Tensor 
     */
    at::Tensor BreakthroughNN::state_to_tensor(Board board){
        
        int bsize = 8;
        // Create a float tensor of zeros
        auto torchopt = torch::TensorOptions()
            .dtype(torch::kFloat32);

        auto out = torch::zeros(
            {2, bsize, bsize}, torchopt
        );

        // convert the board to a tensor
        uint64_t x_board;
        uint64_t o_board;

        // the player to move is the first player
        if(board.to_move == pp::First){
            x_board = board.bbs[0];
            o_board = board.bbs[1];
        } else {
            x_board = board.bbs[1];
            o_board = board.bbs[0];
        }
       
        // iterate over the board and set the values
        for(int i = 0; i < bsize * bsize; i ++){
            int row = i / bsize;
            int col = i % bsize;
            
            if((x_board & 1) != 0){
                out[0][row][col] = 1;
            }

            if((o_board & 1) != 0){
                out[1][row][col] = 1;
            }
            
            x_board = x_board >> 1;
            o_board = o_board >> 1;
        }

        return out;
    }

    /**
     * @brief Convert a map of move ids to probabilities to a policy tensor
     * 
     * @param prob_map 
     * @return at::Tensor 
     */
    at::Tensor BreakthroughNN::move_map_to_policy_tensor(
        move_dist prob_map
    ) {
        at::Tensor policy = torch::zeros({64, 64});

        for(auto &p : prob_map){
            int from = p.first >> 6;
            int to = p.first & 0b111111;

            double prob = p.second;
            policy[from][to] = prob;
        }

        // check that the policy tensor sums to 1
        if(abs(policy.sum().item().toFloat() - 1.) >= 0.001 ){
            throw std::runtime_error("policy tensor does not sum to 1");
        }

        return policy;
    }
}
