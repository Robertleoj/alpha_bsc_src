#include <vector>
#include <string>
#include "../games/move.h"
#include "../games/game.h"
#include "../games/connect4.h"

class Connect4PerfectCompetitor {
public:

    Connect4PerfectCompetitor(
        int num_agents,
        std::string book_path="./7x6.book"
    );

    void update(std::vector<std::string> moves);

    std::vector<std::string> make_and_get_moves();
    std::vector<double> get_results();
private:

    std::vector<std::string> compute_moves(std::vector<std::string>&);
    void evaluate_chunk(
        std::vector<std::string>* positions,
        std::vector<int> * moves
    );

    std::string book_path;
    std::vector<std::stringstream> moves;
    std::vector<double> results;
    std::vector<bool> dead;
    int num_agents;
    int num_dead;
    std::vector<game::IGame *> games;
};