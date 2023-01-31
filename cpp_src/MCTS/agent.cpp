#include "./agent.h"
#include <math.h>
#include <random>
#include <stdexcept>


Agent::Agent(game::IGame & game, pp::Player player)
: game(game), player(player)
{  
    this->tree = new MCTree();
}

void Agent::switch_sides(){ 
    // this->player = this->player == pp::First ? pp::Second: pp::First;
}

Agent::~Agent(){
    if (this->tree != nullptr) {
        delete this->tree;
    }
}

void Agent::update(
    int move_idx
){
    // game move
    auto move = this->tree->root->move_list->begin() + move_idx;
    game.make(move);
    tree->move(move_idx);
}

void Agent::update_tree(int move_idx) {
    this->tree->move(move_idx);
}


double Agent::UCT(MCNode * node, MCNode * childnode){
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

    return (w / n) + c * sqrt(log(N) / n);

}

MCNode * Agent::selection(){

    // If root doesn't exist, create it
    if (this->tree->root == nullptr)
    {
        MCNode *new_node = new MCNode(
            nullptr,
            this->game.moves(),
            -1
        );
        
        this->tree->root = new_node;

        return new_node;
    }

    MCNode * current_node = tree->root;
    
    double max_uct;
    double uct;
    MCNode * next_node = nullptr;
    int best_move;

    MCNode * child_node = nullptr;

    while(true){
        // find best node
        if(current_node->is_terminal){
            return current_node;
        }

        max_uct = -100000;

        auto &move_list = current_node->move_list;

        auto &children = current_node->children;

        for(int i = 0; i < move_list->get_size(); i++){
            if(children[i] == nullptr){

                // update state
                this->game.make(move_list->begin() + i);

                // Make new node 
                MCNode * new_node = new MCNode(
                    current_node,
                    this->game.moves(),
                    i
                );

                children[i] = new_node;

                // return new node
                return new_node;
    
            } else {
                
                // Get uct value
                child_node = children[i];
                uct = UCT(current_node, child_node);

                // Update 
                if(uct > max_uct){
                    max_uct = uct;
                    next_node = child_node;
                    best_move = i;
                }
            }
        }

        this->game.make(move_list->begin() + best_move);

        current_node = next_node;
        next_node = nullptr;
        child_node = nullptr;
    }
}


out::Outcome Agent::simulation(MCNode *selected_node) {
    int cnt = 0;
    // int num_moves;
    int rand_idx;
    out::Outcome wincond;
    
    this->game.push();

    while (true) {
        
        if(this->game.is_terminal()){
            // if counter == 0, update selected node to be terminal
            if(cnt == 0) {
                selected_node->is_terminal = true;
            }
            wincond = this->game.outcome(pp::First);
            break;
        }

        // Select random move
        
        auto ml = game.moves();
        rand_idx = rand() % ml->get_size();
        this->game.make(ml->begin() + rand_idx);
        cnt ++;
    }
    
    this->game.pop();

    return wincond;
};

void Agent::backpropagation(MCNode * node, out::Outcome sim_res){
    while(true){
        node->plays++;
        switch(sim_res){
            case out::Outcome::Tie:
                node->wins += 0.5;
                break;

            case out::Outcome::Win:
                node->wins += this->game.get_to_move() == pp::Second;
                break;

            case out::Outcome::Loss:
                node->wins += this->game.get_to_move() == pp::First;
                break;
            
            default:
                break;
        }

        if (node->parent == nullptr) {
            break;
        } else {
            this->game.retract(node->parent->move_list->begin() + node->idx_in_parent);
            node = node->parent;
        }
    }
}

int Agent::get_current_best_move(){
    // return 0;
    double highest = -10000;
    auto root_children = this->tree->root->children;
    int score;
    int best_move = -1;
    
    for(int i = 0; i < root_children.size(); i++){
        auto child = root_children[i];

        if(child == nullptr){
            continue;
        }

        // score = child->wins / child->plays;
        score = child -> plays;

        if(score > highest){
            highest = score;
            best_move = i;
        }
    }

    if (best_move == -1) {
        throw std::runtime_error("No best move");
    }

    return best_move;
}


game::move_iterator Agent::get_move(int playout_cap){

    int i;
    for(i = 0; i < playout_cap; i++){
        // std::cout << "Playout " << i << std::endl;
        // std::cout << "selection" << std::endl;
        MCNode * selected_node = this->selection();

        // std::cout << "Simulation" << std::endl;
        out::Outcome sim_res = this->simulation(selected_node);

        // std::cout << "Backpropagation" << std::endl;
        this->backpropagation(selected_node, sim_res);
    }

    // Get best move
    printf("Performed %d iterations\n", i);
    auto ret_move = this->get_current_best_move();
    auto mv = this->tree->root->move_list->begin() +  ret_move;
    update(ret_move);
    return mv;
}
