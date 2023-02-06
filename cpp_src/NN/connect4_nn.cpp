#include <string>
#include "./connect4_nn.h"
#include "../base/types.h"


namespace nn{
    
    Connect4NN::Connect4NN(std::string model_path){
        this->net = torch::jit::load(model_path);
        this->net.to(at::kCUDA);
    }

    NNOut Connect4NN::eval_state(Board board) {
        
        auto btensor = this->state_to_tensor(board).cuda();
        // std::cout << btensor.sizes() << std::endl;
        
        std::vector<torch::jit::IValue> inp({btensor});
        
        auto net_out = this->net.forward(inp).toTuple()->elements();
        auto pol_tensor = net_out.at(0).toTensor().cpu().squeeze(0);

        pol_tensor = torch::softmax(pol_tensor, 0);
        auto val_tensor = net_out.at(1).toTensor().cpu().squeeze(0);
        val_tensor = torch::sigmoid(val_tensor);
        
        
        std::map<game::move_id, double> p;

        game::move_id all_moves[7] = {
            1, 2, 3, 4, 5, 6, 7
        };

        // 7 columns in connect4
        // auto random_multinomial = utils::multinomial(7);

        for(int i = 0; i < 7; i++){
            p[all_moves[i]] = pol_tensor[i].item().toDouble();
        }

        return  NNOut {
            p,
            val_tensor.item().toDouble()
        };

    }
    
    at::Tensor Connect4NN::state_to_tensor(Board board){
        
        int rows = 6;
        int cols = 7;
        
        auto torchopt = torch::TensorOptions()
            .dtype(torch::kFloat32);

        auto out = torch::zeros(
            {3, rows, cols}, torchopt
        );
        
        uint64_t x_board = board.bbs[0];
        uint64_t o_board = board.bbs[1];
        
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
        
        bool x_move = board.to_move == pp::First;
        
        if(x_move){
            out[2] += 1;
        }

        return out.unsqueeze(0);
    }


    at::Tensor Connect4NN::visit_count_to_policy_tensor(
        std::map<game::move_id, int> visit_counts
    ) {

        at::Tensor policy = torch::zeros({7});

        // maintain sum to normalize
        double sm = 0;


        for(auto &p : visit_counts){
            // mvoe id is col + 1
            int idx = p.first - 1;
            int cnt = p.second;
            sm += cnt;
            policy[idx] = (double) cnt;
        }


        for(int i = 0; i < 7; i++){
            policy[i] /= sm;
        }

        return policy;
    }
}
