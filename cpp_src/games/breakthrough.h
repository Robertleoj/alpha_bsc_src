//
// Created by Yngvi Björnsson on 7.5.2022.
//
// TO DO:
//  - Not symmetrical move-generation order for White and Black (builtin find and clear)


#ifndef ABG8X8_BREAKTHROUGH_H
#define ABG8X8_BREAKTHROUGH_H
#include <ostream>
#include <string>
#include "../base/types.h"
#include "../base/bitboard.h"
#include "../utils/utils.h"
#include "game.h"
#include "move.h"

namespace games {

    class Breakthrough final : public game::IGame {
        // Breakthrough game.
    public:

        static constexpr int MAX_MOVES = 3 * 2 * FILES;
        using Movelist = game::MovelistPooled<Move,MAX_MOVES>;

        explicit Breakthrough();

        void setup() override;
        bool setup(const struct state_t& state);
        bool setup(const std::string& fen);

        void make(const Move& move);
        void retract(const Move& move);
        void push() override;
        void pop() override;
        bool is_key_unique() const override;

        game::key get_key() const override;

        [[nodiscard]] Outcome outcome(Player pl) const override;
        [[nodiscard]] Outcome outcome() const override;

        int generate(Movelist& move_list) const;

        [[nodiscard]] bool is_terminal() const override;
        [[nodiscard]] bool is_terminal(const Move& move) const;
        [[nodiscard]] bool is_legal() const;
        bool is_legal(const std::string& move_str, mm::Move& move) const;  // NOTE: non-member? and other methods as well.

        [[nodiscard]] Player get_to_move() const override;
        void get_state(state_t& state) const;
        [[nodiscard]] int get_piece_count(Player player) const;

        [[nodiscard]] std::string to_string() const;
        void display(std::ostream& os, const std::string& delimiter = "\n") const override;  // NOTE: non-member.
        bool set(const std::string& pos) override;

        Board get_board() const override;

        // IGame compatibility layer.
        game::MovelistPtr moves() override {
            auto ml_p = Movelist::new_pooled();
            generate(*ml_p);
            return ml_p;
        }

        bool make(game::move_iterator it) override {
            make(*it.as<Move>());
            return true;
        }

        void retract(game::move_iterator it) override {
            retract(*it.as<Move>());
        }

        [[nodiscard]] std::string move_as_str(game::move_iterator it) const override {
            return mm::to_string(*it.as<Move>());
        }

        [[nodiscard]] std::string move_as_str(game::move_id id) const override {
            Move move(id);
            return mm::to_string(move);
        }

        ~Breakthrough();

    private:

        void init();

        template<Player Pl>
        [[nodiscard]] bool is_terminal() const;

        template<Player Pl>
        [[nodiscard]] bool is_terminal(const Move& move) const;

        template<Player Pl>
        int generate(Movelist& move_list) const;

        template<Player Pl>
        [[nodiscard]] int get_piece_count() const;

        struct data_ {
            Player to_move;
            bb::Bitboard bb_pieces[2];
        } m;
        std::vector<data_> stack_;
    };

}

#endif //ABG8X8_BREAKTHROUGH_H
