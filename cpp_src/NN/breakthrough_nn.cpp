#include <string>
#include "./breakthrough_nn.h"
#include "../base/types.h"
#include "../utils/utils.h"
#include <fstream>
#include <cuda_runtime.h>
#include <set>



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
    std::unique_ptr<NNOut> BreakthroughNN::make_nnout_from_tensors(
        at::Tensor pol_tensor, 
        at::Tensor val_tensor
    ){

        throw std::runtime_error("Need to know legal moves in breakthrough");
    }


    /**
     * @brief Create a NNOut from the policy and value tensors
     * 
     * @param pol_tensor 
     * @param val_tensor 
     * @return std::unique_ptr<NNOut> 
     */
    std::unique_ptr<NNOut> BreakthroughNN::make_nnout_from_tensors(
        at::Tensor pol_tensor, 
        at::Tensor val_tensor, 
        std::vector<game::move_id> * legal_moves,
        pp::Player player
    ){
        
        
        nn::move_dist p;

        auto move_id_to_idx = [player](game::move_id id){
            
            int from = id & 0b111111;
            int to = ((id >> 6) & 0b111111);
            if(player == pp::Second){
                from = 63 - from;
                to = 63 - to;
            }

            return std::make_pair(from, to);
        };

        double sm = 0;
        for(auto &id : *legal_moves){
            auto [from, to] = move_id_to_idx(id);
            double val = pol_tensor[from][to].item().toDouble();
            p.emplace(id, val);
            sm += val;
        }

        for(auto &id : *legal_moves){
            p[id] /= sm;
        }

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
    at::Tensor BreakthroughNN::state_to_tensor(
        Board board
    ){
        
        int bsize = 8;
        // Create a float tensor of zeros
        auto torchopt = torch::TensorOptions()
            .dtype(torch::kFloat32);

        auto out = torch::zeros(
            {2, bsize, bsize}, torchopt
        );

        // convert the board to a tensor
        uint64_t x_board = board.bbs[0];
        uint64_t o_board = board.bbs[1];
       
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

        if(board.to_move == pp::Second){
            // flip the board if black
            out = out.flip(0);  // flip channels
            out = out.flip(1);  // flip rows
            out = out.flip(2);  // flip cols
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
        move_dist prob_map,
        pp::Player player
    ) {
        at::Tensor policy = torch::zeros({64, 64});

        for(auto &p : prob_map){
            int from = p.first & 0b111111;
            int to = (p.first >> 6) & 0b111111;

            // flip board if black
            if(player == pp::Second){
                from = 63 - from;
                to = 63 - to;
            }

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
