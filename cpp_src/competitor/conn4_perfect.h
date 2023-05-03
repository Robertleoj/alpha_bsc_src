#include <vector>
#include <string>
#include "../games/move.h"
#include "../games/game.h"
#include "../games/connect4.h"
#include "../conn4_alpha_beta/Solver.hpp"

class Connect4PerfectCompetitor {
public:

    Connect4PerfectCompetitor(
        int num_agents,
        bool random,
        std::string book_path="./7x6.book"
    );

    void update(std::vector<std::string> moves);

    std::vector<std::string> make_and_get_moves();
    std::vector<double> get_results();
private:

    void update_(std::vector<std::string>& moves, bool other_player);
    std::vector<std::string> compute_moves(std::vector<std::string>&);
    void evaluate_chunk(
        std::vector<std::string>* positions,
        std::vector<int> * moves
    );

    int choose_move(std::vector<int> scores);

    bool random;
    std::string book_path;
    std::vector<std::stringstream> moves;
    std::vector<double> results;
    std::vector<bool> dead;
    int num_agents;
    int num_dead;
    std::vector<game::IGame *> games;
};