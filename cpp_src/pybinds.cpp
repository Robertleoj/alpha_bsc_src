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


bool DEBUG = false;

namespace py = pybind11;

std::string model_path(int gen){
    return utils::string_format("./models/%d.pt", gen);
}

class GamePlayer {
public: 
    GamePlayer(int gen) {
        config::initialize();
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
            false
        );
    }

    std::string get_and_make_move(int playout_cap) {

        this->agent->search(
            playout_cap, 
            this->efunc
        );
         
        auto visit_counts = this->agent->root_visit_counts();

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
        game::move_id move_id = mm::from_str(move);
        std::cout << "converted back: " << game->move_as_str(move_id) << std::endl;
        game->make(move_id);
        agent->update_tree(move_id);
    }

private:

    Agent * agent;
    nn::NN * neural_net;
    game::IGame * game;
    eval_f efunc;
};

PYBIND11_MODULE(player, m){
    m.doc() = "Play with agent";

    py::class_<GamePlayer>(m, "GamePlayer")
        .def(py::init<int>())
        .def("get_and_make_move", &GamePlayer::get_and_make_move)
        .def("update", &GamePlayer::update);
}
