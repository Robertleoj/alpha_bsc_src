#include <string>
#include "./connect4_nn.h"
#include "../base/types.h"
#include <fstream>


namespace nn{
    
    Connect4NN::Connect4NN(std::string model_path){
        std::cout << "loading model from " << model_path << std::endl;
        this->net = torch::jit::load(model_path);
        this->net.to(at::kCUDA);
        this->net.eval();
    }

    std::vector<std::unique_ptr<NNOut>> Connect4NN::eval_tensors(std::vector<at::Tensor> & tensors) {
        auto inp_tensor = torch::stack(tensors, 0).cuda();
        // std::cout << "\u001b[36mINP TENSOR SHAPE\u001b[0m" << inp_tensor.sizes() << std::endl;


        std::vector<torch::jit::IValue> inp({inp_tensor});

        auto net_out = this->net.forward(inp).toTuple()->elements();

        auto pol_tensors = net_out.at(0).toTensor();
        pol_tensors = pol_tensors.softmax(1).cpu();

        auto val_tensors = net_out.at(1).toTensor();
        val_tensors = val_tensors.cpu();

        std::vector<std::unique_ptr<NNOut>> out;
        for(int i = 0; i < tensors.size(); i++){
            auto nnout = this->make_nnout_from_tensors(
                pol_tensors[i], val_tensors[i]
            );
            out.push_back(std::move(nnout));
        }
        
        return out;


    }

    std::vector<std::unique_ptr<NNOut>> Connect4NN::eval_states(std::vector<Board> * boards){

        std::vector<at::Tensor> btensors;
        for(auto &b: *boards){
            btensors.push_back(this->state_to_tensor(b));
        }

        return this->eval_tensors(btensors);
    };

    std::unique_ptr<NNOut> Connect4NN::eval_state(Board board) {
        
        auto btensor = this->state_to_tensor(board).unsqueeze(0).cuda();
        // std::cout << btensor.sizes() << std::endl;
        
        std::vector<torch::jit::IValue> inp({btensor});
        
        auto net_out = this->net.forward(inp).toTuple()->elements();
        auto pol_tensor = net_out.at(0).toTensor().cpu().squeeze(0);
        pol_tensor = torch::softmax(pol_tensor, 0);

        auto val_tensor = net_out.at(1).toTensor().cpu().squeeze(0);
        // val_tensor = torch::tanh(val_tensor);

        return std::move(this->make_nnout_from_tensors(pol_tensor, val_tensor));
    }

    std::unique_ptr<NNOut> Connect4NN::make_nnout_from_tensors(at::Tensor pol_tensor, at::Tensor val_tensor){

        std::map<game::move_id, double> p;

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
    
    at::Tensor Connect4NN::state_to_tensor(Board board){
        
        int rows = 6;
        int cols = 7;
        
        auto torchopt = torch::TensorOptions()
            .dtype(torch::kFloat32);

        auto out = torch::zeros(
            {2, rows, cols}, torchopt
        );
        uint64_t x_board;
        uint64_t o_board;
        if(board.to_move == pp::First){
            x_board = board.bbs[0];
            o_board = board.bbs[1];
        } else {
            x_board = board.bbs[1];
            o_board = board.bbs[0];
        }
        
       
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


    at::Tensor Connect4NN::move_map_to_policy_tensor(
        std::map<game::move_id, double> prob_map
    ) {
        /*
            Accepts a normalized map of move ids to probabilities
        */

        at::Tensor policy = torch::zeros({7});

        for(auto &p : prob_map){
            // mvoe id is col + 1
            int idx = p.first - 1;
            double prob = p.second;
            policy[idx] = prob;
        }

        
        if(abs(policy.sum().item().toFloat() - 1.) >= 0.001 ){
            throw std::runtime_error("policy tensor does not sum to 1");
        }

        return policy;
    }
}
