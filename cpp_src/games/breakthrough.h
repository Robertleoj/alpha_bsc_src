#pragma once
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

        std::vector<Move> generate() const;

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
        std::vector<game::move_id> moves() override {
            
            auto moves = generate();
            return std::vector<game::move_id>(moves.begin(), moves.end());
        }

        bool make(game::move_id it) override {
            auto move = Move(it);
            make(move);
            return true;
        }

        void retract(game::move_id id) override {
            retract(Move(id));
        }

        [[nodiscard]] std::string move_as_str(game::move_id mid) const override {
            return mm::to_string(Move(mid));
        }

        ~Breakthrough();

    private:

        void init();

        template<Player Pl>
        [[nodiscard]] bool is_terminal() const;

        template<Player Pl>
        [[nodiscard]] bool is_terminal(const Move& move) const;

        template<Player Pl>
        std::vector<Move> generate() const;

        template<Player Pl>
        [[nodiscard]] int get_piece_count() const;

        struct data_ {
            Player to_move;
            bb::Bitboard bb_pieces[2];
        } m;
        std::vector<data_> stack_;
    };

}
