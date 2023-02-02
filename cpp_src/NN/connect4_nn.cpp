#include <string>
#include "./connect4_nn.h"
#include "../base/types.h"

namespace nn{
    
    Connect4NN::Connect4NN(std::string model_path){
        this->net = torch::jit::load(model_path);
        this->net.to(at::kCUDA);
    }

    NNOut Connect4NN::eval_state(Board board) {
        
        auto btensor = this->board_to_tensor(board).cuda();
        // std::cout << btensor.sizes() << std::endl;
        
        std::vector<torch::jit::IValue> inp({btensor});
        
        auto net_out = this->net.forward(inp);
        
        at::Tensor out_tensor = net_out.toTensor();
        out_tensor = out_tensor.squeeze(0);
        
        std::map<game::move_id, double> p;

        game::move_id all_moves[7] = {
            1, 2, 3, 4, 5, 6, 7
        };

        // 7 columns in connect4
        // auto random_multinomial = utils::multinomial(7);

        for(int i = 0; i < 7; i++){
            p[all_moves[i]] = out_tensor[i].item().toDouble();
        }

        return  NNOut {
            p,
            utils::normalized_double()
        };

    }
    
    torch::Tensor Connect4NN::board_to_tensor(Board board){
        
        int rows = 6;
        int cols = 7;
        
        auto torchopt = torch::TensorOptions()
            .dtype(torch::kFloat32);

        auto out = torch::zeros(
            {3, rows + 1, cols}, torchopt
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
}
