#include "./agent.h"
#include <math.h>
#include <random>
#include <stdexcept>
#include "../hyperparams.h"


Agent::Agent(
    game::IGame * game, 
    pp::Player player, 
    eval_f eval_func
) : game(game), player(player)
{  
    this->eval_func = eval_func;
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

void Agent::update_tree(
    game::move_id move_id
){
    int move_idx = this->tree->root->move_idx_of(move_id);
    tree->move(move_idx);
}


double Agent::PUCT(MCNode * node, MCNode * childnode){
    /*
    PUCT(s, a) = 
        V + c * P(a|s)(\sqrt{N} / (1 + n))
        

    V = evaluations for the child node
    n = number of simulations for child node
    N = simulations for current node
    c = 4
    */
    if(childnode == nullptr){
        throw std::runtime_error("No child node");
    }
    
    double V = childnode->value_approx;
    double n = childnode->plays;
    double P = node->p[childnode->idx_in_parent];
    double N = node->plays;

    double c = hp::PUCT_c; 

    return V + c *P * sqrt(N) / (1 + n);
}

double Agent::outcome_to_value(out::Outcome oc){
    switch(oc){
        case out::Outcome::Loss:
            return 0;
        case out::Outcome::Win:
            return 1;
        case out::Outcome::Tie:
            return 0.5;
        default:
            throw std::runtime_error("Undecided outcome");
    }
}

std::pair<MCNode *, double> Agent::selection(){

    // If root doesn't exist, create it
    if (this->tree->root == nullptr)
    {
        MCNode * new_node = nullptr;
        double v = 0;
        if(game->is_terminal()){
            v = this->outcome_to_value(game->outcome(pp::First));
            new_node = new MCNode(
                nullptr,
                -1,
                v
            );
        } else {
            auto evaluation = this->eval_func(this->game->get_board());

            new_node = new MCNode(
                nullptr,
                this->game->moves(),
                -1,
                *evaluation
            );
            
            v = evaluation->v;
        }
        
        this->tree->root = new_node;

        return std::make_pair(new_node, v);
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
            return std::make_pair(current_node, current_node->value_approx);
        }

        max_uct = -100000;

        auto &move_list = current_node->move_list;

        auto &children = current_node->children;

        for(int i = 0; i < move_list->get_size(); i++){
            if(children[i] == nullptr){

                // update state
                this->game->make(move_list->begin() + i);
                
                MCNode * new_node;
                double v;
                if(game->is_terminal()){
                    v = this->outcome_to_value(game->outcome(pp::First));
                    new_node = new MCNode(
                        current_node,
                        i,
                        v
                    );
                } else {

                    auto evaluation = this->eval_func(this->game->get_board());

                    // Make new node 
                    new_node = new MCNode(
                        current_node,
                        this->game->moves(),
                        i,
                        *evaluation
                    );
                    v = evaluation->v;
                }

                children[i] = new_node;

                // return new node
                return std::make_pair(new_node, v);
    
            } else {
                
                // Get uct value
                child_node = children[i];
                uct = PUCT(current_node, child_node);

                // Update 
                if(uct > max_uct){
                    max_uct = uct;
                    next_node = child_node;
                    best_move = i;
                }
            }
        }

        this->game->make(move_list->begin() + best_move);

        current_node = next_node;
        next_node = nullptr;
        child_node = nullptr;
    }
}


void Agent::backpropagation(MCNode * node, double v){
    while(true){
        node->plays++;
        double val = v;

        if(this->game->get_to_move() == pp::Second){
            val = 1 - v;
        }
        
        node->update_eval(val);

        if (node->parent == nullptr) {
            break;
        } else {
            this->game->retract(node->parent->move_list->begin() + node->idx_in_parent);
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


void Agent::search(int playout_cap){

    int i;

    for(i = 0; i < playout_cap; i++){
        // std::cout << "Playout " << i << std::endl;
        // std::cout << "selection" << std::endl;
        std::pair<MCNode *, double> selection_result = this->selection();
        auto created_node = selection_result.first;
        double v = selection_result.second;

        // std::cout << "Backpropagation" << std::endl;
        this->backpropagation(created_node, v);
    }

    // Get best move
    printf("Performed %d iterations\n", i);
}

std::map<game::move_id, int> Agent::root_visit_counts(){
    return this->tree->root->visit_count_map();
}
