#include "./nn.h"
#include "../utils/utils.h"


namespace nn {
    NN::NN(std::string model_path){
        std::cout << "loading model from " << model_path << std::endl;
        this->net = torch::jit::load(model_path);
        this->net.to(at::kCUDA);
        this->net.eval();
    }

    c10::ivalue::TupleElements NN::run_batch(at::Tensor inp_tensor){
        if(!inp_tensor.is_cuda()){
            inp_tensor = inp_tensor.cuda();
        }

        // Create the input value for the network
        std::vector<torch::jit::IValue> inp({inp_tensor});

        // Get the output from the network
        auto net_out = this->net.forward(inp).toTuple()->elements();

        // cudaFree(inp_tensor.data_ptr());

        return net_out;
    }

    std::vector<std::unique_ptr<NNOut>> NN::eval_batch(at::Tensor inp_tensor){


        // Create the input value for the network
        std::vector<torch::jit::IValue> inp({inp_tensor});

        // Get the output from the network
        auto net_out = this->net.forward(inp).toTuple()->elements();

        // Get the policy and value tensors
        auto pol_tensors = net_out.at(0).toTensor();
        pol_tensors = this->pol_softmax(pol_tensors).cpu();

        auto val_tensors = net_out.at(1).toTensor();
        val_tensors = val_tensors.cpu();

        // Create the output

        std::vector<std::unique_ptr<NNOut>> out;
        for(int i = 0; i < inp_tensor.size(0); i++){
            auto nnout = this->make_nnout_from_tensors(
                pol_tensors[i], val_tensors[i]
            );
            out.push_back(std::move(nnout));
        }
        
        return out;
    }


    /**
     * @brief Evaluate a single state
     * 
     * @param board 
     * @return std::unique_ptr<NNOut> 
     */
    std::unique_ptr<NNOut> NN::eval_state(Board board) {
        
        // Convert the state to a tensor
        auto btensor = this->state_to_tensor(board).unsqueeze(0).cuda();
        
        // Create the input value for the network
        std::vector<torch::jit::IValue> inp({btensor});
        
        // Get the output from the network
        auto net_out = this->net.forward(inp).toTuple()->elements();

        // Get the policy and value tensors
        auto pol_tensor = net_out.at(0).toTensor().cpu();
        pol_tensor = this->pol_softmax(pol_tensor).squeeze(0);

        auto val_tensor = net_out.at(1).toTensor().cpu().squeeze(0);

        // Create the output
        return std::move(this->make_nnout_from_tensors(pol_tensor, val_tensor));
    }

    at::Tensor NN::prepare_batch(std::vector<at::Tensor>& tensors){
        auto inp_tensor = torch::stack(tensors, 0);

        // if(!inp_tensor.is_cuda()){
        //     inp_tensor = inp_tensor.cuda();
        // }
        return inp_tensor;
    }


    std::vector<std::unique_ptr<NNOut>> NN::net_out_to_nnout(
        at::Tensor pol_tensors, 
        at::Tensor val_tensors,
        std::vector<std::vector<game::move_id> *> legal_moves,
        std::vector<pp::Player> *to_move
    ){
        // Get the policy and value tensors
        // auto pol_tensors = net_out.at(0).toTensor();
        pol_tensors = this->pol_softmax(pol_tensors).cpu();

        // auto val_tensors = net_out.at(1).toTensor();
        val_tensors = val_tensors.cpu();

        // Create the output

        // std::vector<std::tuple<
        //     at::Tensor, 
        //     at::Tensor, 
        //     std::vector<game::move_id> *
        // >> inp;

        // for(int i = 0; i < pol_tensors.size(0); i++){
        //     inp.push_back(std::make_tuple(
        //         pol_tensors[i], 
        //         val_tensors[i],
        //         legal_moves[i]
        //     ));
        // }

        // auto pool = utils::ThreadPool<
        //     std::tuple<at::Tensor, at::Tensor, std::vector<game::move_id> *>,
        //     std::unique_ptr<nn::NNOut>
        // >(10);       

        // // auto t = utils::Timer(); t.start();
        // auto ret = pool.map(
        //     inp,
        //     [this](std::tuple<at::Tensor, at::Tensor, std::vector<game::move_id> *> inp){
        //         return std::move(this->make_nnout_from_tensors(
        //             std::get<0>(inp), 
        //             std::get<1>(inp),
        //             std::get<2>(inp)
        //         ));
        //     }
        // );
        // t.stop();t.print();

        // auto t = utils::Timer(); t.start();
        std::vector<std::unique_ptr<NNOut>> ret;

        for(int i= 0; i < pol_tensors.size(0); i++){
            ret.push_back(this->make_nnout_from_tensors(
                pol_tensors[i], val_tensors[i], legal_moves[i], (*to_move)[i]
            ));
        }
        // t.stop();t.print();

        return ret;
    }


}