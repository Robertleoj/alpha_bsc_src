#include "./agent.h"
#include <math.h>
#include <random>
#include <stdexcept>
#include "../utils/utils.h"
#include "../config/config.h"


Agent::Agent(
    game::IGame * game, 
    pp::Player player, 
    eval_f eval_func,
    bool apply_noise
) : game(game), player(player)
{  
    this->eval_func = eval_func;
    this->tree = new MCTree();
    this->use_dirichlet_noise = apply_noise;
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
    tree->move(move_id);

    if(!this->game->is_terminal() && this->tree->root != nullptr){
        if(this->use_dirichlet_noise) {
            auto dir_noise = utils::dirichlet_dist(
                config::hp["dirichlet_alpha"].get<double>(), 
                this->tree->root->legal_moves.size()
            );
            this->tree->root->add_noise(dir_noise);
        }
    }
}


double Agent::PUCT(MCNode * node, game::move_id move){
    /*
    PUCT(s, a) = 
        V + c * P(a|s)(\sqrt{N} / (1 + n))
        

    V = evaluations for the child node
    n = number of simulations for child node
    N = simulations for current node
    c = hp
    */

    double P = node->p_map[move];
    double c = config::hp["PUCT_c"].get<double>(); 

    MCNode * childnode = node->children[move];

    if(childnode == nullptr){
        return c * P;
        // throw std::runtime_error("No child node");
    }
    
    double V = childnode->value_approx;
    double n = childnode->plays;
    double N = node->plays;


    return V + c *P * sqrt(N) / (1 + n);
}

double Agent::outcome_to_value(out::Outcome oc){
    switch(oc){
        case out::Outcome::Loss:
            return -1;
        case out::Outcome::Win:
            return 1;
        case out::Outcome::Tie:
            return 0;
        default:
            throw std::runtime_error("Undecided outcome");
    }
}

double Agent::eval_for_player(double v, pp::Player player){
    if(player == pp::First){
        return v;
    } else {
        return -v;
    }
}

double Agent::switch_eval(double v) {
    return -v;
}

std::pair<MCNode *, double> Agent::selection(){

    // If root doesn't exist, create it
    if (this->tree->root == nullptr)
    {
        MCNode * new_node = nullptr;
        double v = 0;
        if(game->is_terminal()){
            v = this->outcome_to_value(game->outcome());
            new_node = new MCNode(
                nullptr,
                -1
            );
        } else {
            auto evaluation = this->eval_func(this->game->get_board());

            new_node = new MCNode(
                nullptr,
                this->game->moves(),
                -1,
                *evaluation
            );
            
            if(this->use_dirichlet_noise){
                auto dir_dist = utils::dirichlet_dist(config::hp["dirichlet_alpha"].get<double>(), new_node->legal_moves.size());
                new_node->add_noise(dir_dist);
            }
            
            v = evaluation->v;
        }
        
        this->tree->root = new_node;

        return std::make_pair(new_node, this->switch_eval(v));
    }

    MCNode * current_node = tree->root;
    
    double max_uct;
    double uct;
    MCNode * next_node = nullptr;
    game::move_id best_move;

    // MCNode * child_node = nullptr;

    while(true){
        // find best node
        if(current_node->is_terminal){
            return std::make_pair(current_node, current_node->value_approx);
        }

        max_uct = -100000;

        auto &legal_moves = current_node->legal_moves;

        auto &children = current_node->children;

        for(auto mv: legal_moves){
            uct = PUCT(current_node, mv);

            if(uct > max_uct){
                max_uct = uct;
                next_node = children[mv];
                best_move = mv;
            }
        }


        game->make(best_move);

        if(next_node == nullptr){
            // create new node
            MCNode * new_node = nullptr;
            double v;


            if(game->is_terminal()){
                v = this->outcome_to_value(game->outcome());

                new_node = new MCNode(
                    current_node,
                    best_move
                );
            } else {

                auto evaluation = this->eval_func(this->game->get_board());

                // Make new node 
                new_node = new MCNode(
                    current_node,
                    this->game->moves(),
                    best_move,
                    *evaluation
                );
                v = evaluation->v;
            }


            children[best_move] = new_node;

            return std::make_pair(new_node, this->switch_eval(v));
        } else {
            //continue selection
            current_node = next_node;
            next_node = nullptr;
        }
    }
}


void Agent::backpropagation(MCNode * node, double v){
    // bool first = true;

    while(true){

        // if(!first){
        node->update_eval(v);
        // }
        // first = false;

        if (node->parent == nullptr) {
            break;
        } else {
            this->game->retract(node->move_from_parent);
            node = node->parent;
            v = this->switch_eval(v);
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
    // printf("Performed %d iterations\n", i);
}

std::map<game::move_id, int> Agent::root_visit_counts(){
    return this->tree->root->visit_count_map();
}
