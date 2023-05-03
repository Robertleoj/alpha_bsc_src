#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "./MCTS/agent.h"
#include "./NN/nn.h"
#include "./NN/connect4_nn.h"
#include "./NN/breakthrough_nn.h"
#include "./global.h"
#include "./base/types.h"
#include "./games/game.h"
#include "./games/breakthrough.h"
#include "./games/move.h"
#include "./config/config.h"
#include "./vanilla_MCTS/v_agent.h"
#include "./competitor/competitor.h"
#include "./competitor/conn4_perfect.h"


bool DEBUG = false;


std::string model_path(int gen){
    return utils::string_format("./models/%d.pt", gen);
}


class GamePlayer {
public: 
    GamePlayer(int gen, int playouts) {
        config::initialize();
        this->playouts = playouts;
        this->game = new games::Breakthrough();
        std::string mdl_path = model_path(gen);
        this->neural_net = new nn::BreakthroughNN(mdl_path);

        this->efunc = [this](
            Board b, 
            std::vector<game::move_id> * legal_moves
        ) {
            return this->neural_net->eval_state(b, legal_moves);
        };

        this->agent = new Agent(
            this->game,
            false,
            false
        );
    }

    std::string get_and_make_move() {

        this->agent->search(
            this->playouts, 
            this->efunc
        );
         
        auto visit_counts = this->agent->root_visit_counts();
        auto prior = this->agent->tree->root->p_map;
        
        // for(auto &p : prior) {
        //     std::cout << p.first << " " << p.second << std::endl;
        // }


        // create a set with a lambda function to sort the values
        auto value_comparator = [](const auto& a, const auto& b) { return a.second < b.second; };

        std::set<std::pair<game::move_id, double>, decltype(value_comparator)> stuff(prior.begin(), prior.end(), value_comparator);

        // print the map in sorted order of values
        std::cout << "Prior:" << std::endl;
        for (const auto& pair : stuff) {
            std::cout << this->game->move_as_str(pair.first) << ": " << pair.second << std::endl;
        }


        double val = this->agent->tree->root->value_approx;
        std::cout << "Value: " << val << std::endl;


        game::move_id best_move;
        int best_visit_count = -1;

        for(auto &p : visit_counts) {
            if(p.second > best_visit_count){
                best_move = p.first;
                best_visit_count = p.second;
            }
        }

        std::string move_str = game->move_as_str(best_move);

        game->make(best_move);
        agent->update_tree(best_move);

        return move_str;

    }

    void update(std::string move){
        game::move_id move_id = this->game->move_from_str(move);
        game->make(move_id);
        agent->update_tree(move_id);
    }

private:

    Agent * agent;
    nn::NN * neural_net;
    game::IGame * game;
    eval_f efunc;
    int playouts;
};

class VanillaPlayer {
public:
    VanillaPlayer(int playouts, std::string game_name="breakthrough") {

        if(game_name == "breakthrough") {
            this->game = new games::Breakthrough();
        } 
        else if (game_name == "connect4") {
            this->game = new games::Connect4();
        }

        this->playouts = playouts;

        this->agent = new VAgent(
            this->game
        );
    }

    std::string get_and_make_move(){
        game::move_id move = this->agent->get_move(this->playouts);
        this->game->make(move);
        this->agent->update_tree(move);

        std::string move_str = game->move_as_str(move);
        return move_str;
    }

    void update(std::string move){
        game::move_id move_id = this->game->move_from_str(move);
        game->make(move_id);
        agent->update_tree(move_id);
    }

private:
    VAgent * agent;
    game::IGame * game;
    int playouts;
};



namespace py = pybind11;

PYBIND11_MODULE(player, m){
    m.doc() = "Play with agent";

    py::class_<GamePlayer>(m, "GamePlayer")
        .def(py::init<int, int>())
        .def("get_and_make_move", &GamePlayer::get_and_make_move)
        .def("update", &GamePlayer::update);

    py::class_<VanillaPlayer>(m, "VanillaPlayer")
        .def(py::init<int, std::string>())
        .def(py::init<int>())
        .def("get_and_make_move", &VanillaPlayer::get_and_make_move)
        .def("update", &VanillaPlayer::update);

    py::class_<Competitor>(m, "Competitor")
        .def(py::init<std::string, std::string, int, int, int>())
        .def("update", &Competitor::update)
        .def("get_results", &Competitor::get_results)
        .def("make_and_get_moves", &Competitor::make_and_get_moves);

    py::class_<Connect4PerfectCompetitor>(m, "Connect4PerfectCompetitor")
        .def(py::init<int, std::string>())
        .def(py::init<int>())
        .def("update", &Connect4PerfectCompetitor::update)
        .def("get_results", &Connect4PerfectCompetitor::get_results)
        .def("make_and_get_moves", &Connect4PerfectCompetitor::make_and_get_moves);
}
