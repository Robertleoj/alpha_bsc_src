#include "./v_agent.h"
#include <math.h>
#include <random>
#include <stdexcept>


VAgent::VAgent(
    game::IGame * game
) : game(game) {  
    this->tree = new VMCTree();
}


VAgent::~VAgent(){
    if (this->tree != nullptr) {
        delete this->tree;
    }
}

void VAgent::update_tree(game::move_id move_id) {
    this->tree->move(move_id);
}


double VAgent::UCT(VMCNode * node, VMCNode * childnode){
    /*
    UCT(node, move) = 
        (w / n) + c * sqrt(ln(N) / n)
    w = wins for child node after move
    n = number of simulations for child node
    N = simulations for current node
    c = sqrt(2)
    */
    if(childnode == nullptr){
        throw std::runtime_error("No child node");
    }


    double w = childnode->wins;
    double n = childnode->plays;
    double N = node->plays;
    double c = sqrt(2);

    if(n == 0){
        throw std::runtime_error("No plays"); 
    }

    return (w / n) + c * sqrt(log(N) / n);

}

VMCNode * VAgent::selection(){

    // If root doesn't exist, create it
    if (this->tree->root == nullptr)
    {
        VMCNode *new_node = new VMCNode(
            nullptr,
            0,
            this->game->moves()
        );
        
        this->tree->root = new_node;

        return new_node;
    }

    VMCNode * current_node = tree->root;
    
    double max_uct;
    double uct;
    VMCNode * next_node = nullptr;
    game::move_id best_move;

    while(true){
        // find best node
        if(current_node->is_terminal){
            return current_node;
        }

        max_uct = -100000;


        for(auto ch : current_node->children){
            auto [move_id, child]  = ch;

            if(child == nullptr){

                // update state
                this->game->make(move_id);

                // Make new node 
                VMCNode * new_node = new VMCNode(
                    current_node,
                    move_id,
                    this->game->moves()
                );

                current_node->children[move_id] = new_node;

                // return new node
                return new_node;
    
            } else {
                
                // Get uct value
                uct = UCT(current_node, child);

                // Update 
                if(uct > max_uct){
                    max_uct = uct;
                    next_node = child;
                    best_move = move_id;
                }
            }
        }

        this->game->make(best_move);

        current_node = next_node;
        next_node = nullptr;
    }
}


out::Outcome VAgent::simulation(VMCNode *selected_node) {
    int cnt = 0;
    // int num_moves;
    int rand_idx;
    out::Outcome wincond;
    
    this->game->push();

    while (true) {
        
        if(this->game->is_terminal()){
            // if counter == 0, update selected node to be terminal
            if(cnt == 0) {
                selected_node->is_terminal = true;
            }
            wincond = this->game->outcome(pp::First);
            break;
        }

        // Select random move
        
        auto ml = game->moves();
        rand_idx = rand() % ml.size();
        this->game->make(ml[rand_idx]);
        cnt ++;
    }
    
    this->game->pop();

    return wincond;
};

void VAgent::backpropagation(VMCNode * node, out::Outcome sim_res){
    while(true){
        node->plays++;
        switch(sim_res){
            case out::Outcome::Tie:
                node->wins += 0.5;
                break;

            case out::Outcome::Win:
                node->wins += this->game->get_to_move() == pp::Second;
                break;

            case out::Outcome::Loss:
                node->wins += this->game->get_to_move() == pp::First;
                break;
            
            default:
                break;
        }

        if (node->parent == nullptr) {
            break;
        } else {
            this->game->retract(node->move_from_parent);
            node = node->parent;
        }
    }
}

game::move_id VAgent::get_current_best_move(){
    // return 0;
    int highest = -10000;
    auto root_children = this->tree->root->children;
    int score;
    bool invalid = true;

    game::move_id best_move = 0;
    
    for(auto &ch : root_children){
        auto [id, child] = ch;

        if(child == nullptr){
            continue;
        }

        score = child -> plays;

        if(score > highest){
            highest = score;
            best_move = id;
            invalid = false;
        }
    }

    if (invalid) {
        throw std::runtime_error("No best move");
    }

    return best_move;
}


game::move_id VAgent::get_move(int playout_cap){

    int i;
    for(i = 0; i < playout_cap; i++){
        // std::cout << "Playout " << i << std::endl;
        // std::cout << "selection" << std::endl;
        VMCNode * selected_node = this->selection();

        // std::cout << "Simulation" << std::endl;
        out::Outcome sim_res = this->simulation(selected_node);

        // std::cout << "Backpropagation" << std::endl;
        this->backpropagation(selected_node, sim_res);
    }

    // Get best move
    printf("Performed %d iterations\n", i);
    return this->get_current_best_move();
}