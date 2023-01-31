#pragma once
#include "./nn.h"
#include "../utils/utils.h"


namespace nn{
    class Connect4NN: public NN{
    public:
        NNOut eval_state(Board board) override {
            
            std::map<game::move_id, double> p;

            game::move_id all_moves[7] = {
                1, 2, 3, 4, 5, 6, 7
            };

            // 7 columns in connect4
            auto random_multinomial = utils::multinomial(7);

            for(int i = 0; i < 7; i++){
                p[all_moves[i]] = random_multinomial[i];
            }

            return  NNOut {
                p,
                utils::normalized_double()
            };

        }
    };
}

