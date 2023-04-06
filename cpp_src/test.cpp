#include "./games/breakthrough.h"
#include "./games/game.h"
#include "./global.h"
#include "./base/types.h"


const bool DEBUG = true;

template <typename T>
T get_rand(std::vector<T> vec){
    return vec[rand() % vec.size()];
}


int main(int argc, char *argv[]) {
    game::IGame * game = new games::Breakthrough();

    game->display(std::cout);
    std::cout << std::endl;


    while(!game->is_terminal()){
        auto moves = game->moves();
        auto move_id = get_rand(moves);

        auto move_str = game->move_as_str(move_id);

        std::string to_move = game->get_to_move() == pp::First ? "First" : "Second";

        std::cout << "To move: " << to_move << std::endl;

        std::cout << "Move: " << move_str << std::endl;
        std::cout << "Move ID " << move_id << std::endl;

        // get from and to
        int from = move_id & 0b111111;
        int to = (move_id >> 6) & 0b111111;

        std::cout << "From: " << from << std::endl;
        std::cout << "To: " << to << std::endl;

        int flipped_from = 63 - from;
        int flipped_to = 63 - to;
        game::move_id flipped_move = (flipped_to << 6) | flipped_from;

        auto flipped_move_str = game->move_as_str(flipped_move);

        std::cout << "Flipped Move: " << flipped_move_str << std::endl;

        std::cout << "Flipped Move ID" << flipped_move << std::endl;

        std::cout << "Flipped From: " << flipped_from << std::endl;

        std::cout << "Flipped To: " << flipped_to << std::endl;

        game->make(move_id);

        game->display(std::cout);
        std::cout << std::endl;

        std::string inp;
        std::cout << "Press enter to continue..." << std::flush;
        std::getline(std::cin, inp);
    }
}