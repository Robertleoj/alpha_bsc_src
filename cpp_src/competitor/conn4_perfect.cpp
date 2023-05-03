#include "./conn4_perfect.h"
#include <map>

int sign(int x){
    if(x > 0){
        return 1;
    } else if(x < 0){
        return -1;
    } else {
        return 0;
    }
}

void Connect4PerfectCompetitor::evaluate_chunk(
    std::vector<std::string>* positions, 
    std::vector<int> *moves
){
    using namespace GameSolver;
    Solver solver;

    solver.loadBook(book_path);
    
    for(auto &s: *positions){
        Position P;

        if(P.play(s) != s.size()) {
            throw std::runtime_error("Invalid move sequence");
        } else {
            std::vector<int> scores = solver.analyze(P, false);
            // map scores to -1, 0, 1
            int best_score = -2;

            for(int i = 0; i < scores.size(); i++){

                if(scores[i] == -1000){
                    continue;
                }

                scores[i] = sign(scores[i]);
                if(scores[i] > best_score){
                    best_score = scores[i];
                }
            }

            // choose random move from best moves
            std::vector<int> best_moves;
            for(int i = 0; i < scores.size(); i++){
                if(scores[i] == best_score){
                    best_moves.push_back(i);
                }
            }

            if(best_moves.size() == 0){
                throw std::runtime_error("No best moves");
            }

            int move = best_moves[rand() % best_moves.size()];
            moves->push_back(move + 1);
        }
    }
}

double outcome_to_value(out::Outcome oc){
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


std::vector<std::string> Connect4PerfectCompetitor::compute_moves(
    std::vector<std::string>& move_strings
){
    using namespace GameSolver;

    int THREADS = std::min(
        (int) std::thread::hardware_concurrency(),
        std::max((int) move_strings.size() / 10, 1)
    );

    std::vector<int> moves[THREADS];

    std::vector<int> all_moves;

    std::vector<std::thread> threads;

    std::vector<std::string> chunks[THREADS];

    int positions_per_thread = move_strings.size() / THREADS;

    // chunk the positions into THREAD chunks
    for(int i = 0; i < move_strings.size(); i++){

        int idx = i / positions_per_thread;

        if(idx >= THREADS){
            idx = THREADS - 1;
        }

        chunks[idx].push_back(move_strings[i]);
    }

    for(int i = 0; i < THREADS; i++){
        auto chunk_ptr = &chunks[i];
        auto move_ptr = &moves[i];

        threads.push_back(std::thread([this, chunk_ptr, move_ptr](){
            this->evaluate_chunk(chunk_ptr, move_ptr);
        }));
    }

    for(int i = 0; i < THREADS; i++){
        threads[i].join();
    }

    for(int i = 0; i < THREADS; i++){
        for(auto v: moves[i]){
            all_moves.push_back(v);
        }
    }

    // map the moves to strings
    std::vector<std::string> all_move_strings(all_moves.size());

    for(int i = 0; i < all_moves.size(); i++){
        all_move_strings[i] = std::to_string(all_moves[i]);
    }

    return all_move_strings;
}

Connect4PerfectCompetitor::Connect4PerfectCompetitor(
    int num_agents,
    std::string book_path
) {
    this->num_agents = num_agents;
    this->num_dead = 0;
    this->book_path = book_path;
    this->moves = std::vector<std::stringstream>(num_agents);
    this->results = std::vector(num_agents, 0.0);
    this->dead = std::vector(num_agents, false);
    for(int i = 0; i < num_agents; i++) {
        this->games.push_back(new games::Connect4());
    }
}

void Connect4PerfectCompetitor::update(
    std::vector<std::string> moves
) {
    for(int i = 0; i < moves.size(); i++){
        auto move = moves[i];
        if(move == ""){
            continue;
        }

        if(!this->dead[i]) {
            game::move_id move_id = this->games[i]->move_from_str(move);
            this->games[i]->make(move_id);
            this->moves[i] << move;

            if(this->games[i]->is_terminal()) {

                auto outcome = this->games[i]->outcome(
                    this->games[i]->get_to_move()
                );

                this->results[i] = outcome_to_value(outcome);

                this->dead[i] = true;
                delete this->games[i];
                this->num_dead++;
            } 
        } else {
            throw std::runtime_error("Game is over");
        }
    }
}

std::vector<std::string> Connect4PerfectCompetitor::make_and_get_moves() {

    std::map<int, int> indices;

    std::vector<std::string> positions_to_compute;

    for(int i =0; i < this->num_agents; i++) {
        if(!this->dead[i]) {
            indices[i] = positions_to_compute.size();
            positions_to_compute.push_back(this->moves[i].str());
        }
    }

    auto computed_moves = this->compute_moves(positions_to_compute);

    std::vector<std::string> moves(this->num_agents);
    for(int i = 0; i < this->num_agents; i++) {
        if(!this->dead[i]){
            moves[i] = computed_moves[indices[i]];
        }
    }
    
    this->update(moves);
    return moves;
}

std::vector<double> Connect4PerfectCompetitor::get_results(){
    return this->results;
}