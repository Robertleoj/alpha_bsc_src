#include <string>
#include "./connect4_nn.h"
#include "../base/types.h"
#include <fstream>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>



namespace nn{
    
    /**
     * @brief Constructor
     * 
     * @param model_path 
     */
    Connect4NN::Connect4NN(std::string model_path): NN(model_path){}

    at::Tensor Connect4NN::pol_softmax(at::Tensor pol_tensor){
        return pol_tensor.softmax(1);
    }

    /**
     * @brief Create a NNOut from the policy and value tensors
     * 
     * @param pol_tensor 
     * @param val_tensor 
     * @return std::unique_ptr<NNOut> 
     */
    std::unique_ptr<NNOut> Connect4NN::make_nnout_from_tensors(at::Tensor pol_tensor, at::Tensor val_tensor){

        move_dist p;

        game::move_id all_moves[7] = {
            1, 2, 3, 4, 5, 6, 7
        };

        // 7 columns in connect4

        for(int i = 0; i < 7; i++){
            p[all_moves[i]] = pol_tensor[i].item().toDouble();
        }

        return std::unique_ptr<NNOut>(new NNOut{
            p,
            val_tensor.item().toDouble()
        });
    }

    std::unique_ptr<NNOut> Connect4NN::make_nnout_from_tensors(
        at::Tensor policy_tensor,
        at::Tensor value_tensor,
        std::vector<game::move_id> * legal_moves,
        pp::Player to_move
    ) {
        return make_nnout_from_tensors(policy_tensor, value_tensor);
    }

    /**
     * @brief Convert a state to a tensor
     * 
     * @param board 
     * @return at::Tensor 
     */
    at::Tensor Connect4NN::state_to_tensor(Board board){
        
        int rows = 6;
        int cols = 7;
        
        // Create a float tensor of zeros
        auto torchopt = torch::TensorOptions()
            .dtype(torch::kFloat32);
        auto out = torch::zeros(
            {2, rows, cols}, torchopt
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
        for(int i = 0; i < rows * cols; i ++){
            int row = i / cols;
            int col = i % cols;
            
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
    at::Tensor Connect4NN::move_map_to_policy_tensor(
        move_dist prob_map,
        pp::Player to_move
    ) {
        at::Tensor policy = torch::zeros({7});

        for(auto &p : prob_map){
            // mvoe id is col + 1
            int idx = p.first - 1;
            double prob = p.second;
            policy[idx] = prob;
        }

        // check that the policy tensor sums to 1
        if(abs(policy.sum().item().toFloat() - 1.) >= 0.001 ){
            throw std::runtime_error("policy tensor does not sum to 1");
        }

        return policy;
    }
}
