#pragma once

#include <ostream>
#include <assert.h>
#include "game.h"
#include "../base/bitboard.h"
#include "../base/board.h"
#include "../utils/colors.h"



namespace games {

    class Connect4 final : public game::IGame {
    public:

        static const int NUM_COLS = 7;
        static const int NUM_ROWS = 6;

        static constexpr bb::Bitboard file(int file) {
            assert(NUM_ROWS==6);
            // Need to change if num rows changes.
            return bit(file) 
                | bit(file+NUM_COLS) 
                | bit(file+2*NUM_COLS) 
                | bit(file+3*NUM_COLS) 
                | bit(file+4*NUM_COLS) 
                | bit(file+5*NUM_COLS);
        }

        static constexpr bb::Bitboard bit(int bit) { 
            return 1ULL << bit; 
        }

        static constexpr unsigned int order[] = {
            3, 2, 4, 1, 5, 0, 6
        };  // Change if num cols changes.


        struct Move final {
            explicit Move(unsigned int c = 0) : col(c) {}

            explicit Move(game::move_id id) : col(id) {}

            unsigned int col;

            [[nodiscard]] operator game::move_id() const {
                return (game::move_id) col; 
            }

            bool operator==(const Move& rhs) { 
                return this->col == rhs.col; 
            }
        };

        explicit Connect4() {
            clear();
        }

        ~Connect4() {
        }

        void setup() override {
            clear();
        }

        bool set(const std::string& pos) override {
            clear();
            for (auto c : pos) {
                auto col = c - '1';
                if (col < 0 || col >= NUM_COLS) {
                    return false;
                }
                if (m.height[col] >= NUM_ROWS) {
                    return false;
                }

                bb::Bitboard move = 1ULL << square(
                    col, m.height[col]++
                );

                m.board[m.counter++ & 1] ^= move;
            }
            return true;
        }

        game::move_id move_from_str(std::string move) const override {
            return std::stoi(move);
        }

        std::vector<game::move_id> moves() override {

            std::vector<game::move_id> moves;

            for (unsigned int i = 0; i < NUM_COLS; i++) {
                auto col = order[i];
                if (m.height[col] < NUM_ROWS) {
                    moves.push_back(col+1);
                }
            }
            return moves;

        }

        bool make(game::move_id mid) override {
            auto col = mid - 1;

            if(col >= 7){
                throw std::runtime_error("stupid col: " + std::to_string(col));
            }
            
            if(m.height[col] >= NUM_ROWS){
                this->display(std::cout);

                std::cout << std::endl;

                std::cout << "Played column" 
                          << col 
                          << std::endl;

                throw std::runtime_error("invalid move");
            }

            bb::Bitboard move = 1ULL << square(
                col, m.height[col]++
            );

            m.board[m.counter++ & 1] ^= move;

            // m.board[m.counter++ & 1] += move;
            return true;
        }

        void retract(game::move_id mid) override {
            auto col = mid - 1;

            bb::Bitboard move = 1ULL << square(
                col, --m.height[col]
            );

            m.board[--m.counter & 1] ^= move;
        }

        [[nodiscard]] std::string move_as_str(
            game::move_id mid
        ) const override {
            return std::to_string(mid);
        }

        [[nodiscard]] bool is_terminal(
        ) const override {
            return (m.counter == (NUM_COLS*NUM_ROWS)) || is_win(m.board[(m.counter+1) & 1]);
        }

        [[nodiscard]] Outcome outcome(
            Player pl
        ) const override {

            if (is_terminal()) {

                if (is_win(m.board[other(pl)])) {
                    return Outcome::Loss;
                }
                else if (is_win(m.board[pl])) {
                    return Outcome::Win;
                }
                else { 
                    return Outcome::Tie; 
                }

            } else { 
                return Outcome::Undecided; 
            }
        }

        [[nodiscard]] Outcome outcome() const override {
            return outcome(get_to_move());
        }

        [[nodiscard]] Player get_to_move() const override
        {
            return (m.counter & 1) ? Second : First;
        }

        void push() override {
            stack_.push_back(m);
        }

        void pop() override {
            if (!stack_.empty()) {
                m = stack_.back();
                stack_.pop_back();
            }
        }

        bool is_key_unique() const override {
            return true;
        }

        game::key get_key() const override {
            game::key k = m.board[1];
            return (k << 64) | m.board[0];
        }

        void display(
            std::ostream& os, 
            const std::string& delimiter = IGame::default_delimiter
        ) const override {

            for (int row = NUM_ROWS-1; row >= 0; --row) {

                os << '|';

                for (int col = 0; col < NUM_COLS; ++col) {

                    int sqr = square(col, row);

                    if (
                        (m.board[0] & (1ULL << sqr)) 
                        != 0ULL
                    ) {
                        // blue X
                        os << colors::BLUE << "X" << colors::RESET;
                    }

                    else if (
                        (m.board[1] & (1ULL << sqr)) 
                        != 0ULL
                    ) {
                        // green O
                        os << colors::GREEN << "O" << colors::RESET;
                    }
                    else { os << ' '; }
                    os << '|';
                }
                os << delimiter;
            }
            os<< m.counter; //<< delimiter;
        }

        Board get_board() const override
        {
            Board board;
            board.to_move = get_to_move();
            board.bbs.push_back(m.board[0]);
            board.bbs.push_back(m.board[1]);
            return board;
        }

    private:

        inline static int square(int col, int row) { 
            return row * NUM_COLS + col; 
        }
        // inline static int column(int square) { return square % NUM_COLS; }
        // inline static int row(int square) { return square / NUM_COLS; }

        static bool is_win(bb::Bitboard bb_player) {
            static constexpr bb::Bitboard leftmost = ~file(0);

            static constexpr bb::Bitboard rightmost = ~file(NUM_COLS-1);

            static constexpr bb::Bitboard leftmost3 =~(
                file(0) 
                | file(1) 
                | file(2)
            );

            static constexpr bb::Bitboard rightmost3 = ~(
                file(NUM_COLS-1) 
                | file(NUM_COLS-2) 
                | file(NUM_COLS-3)
            );

            static constexpr bb::Bitboard masks1[4] = {
                rightmost, bb::full, leftmost, rightmost
            };

            static constexpr bb::Bitboard masks2[4] = {
                rightmost3, bb::full, leftmost3, rightmost3
            };

            static constexpr int directions[4] = {
                1, NUM_COLS, NUM_COLS-1, NUM_COLS+1
            };

            bb::Bitboard bb;

            for (auto i=0; i<4; ++i) {

                auto d = directions[i];

                bb = bb_player 
                    & (bb_player >> d) 
                    & masks1[i];

                if (bb & (bb >> (2 * d)) & masks2[i]) {
                    return true;
                }

            }
            return false;
        }

        [[nodiscard]] static Player other(Player player) {
            return (player == First) ? Second : First;
        }

        void clear() {
            m.board[0] = m.board[1] = 0ULL;
            m.counter = 0;

            for (int & col : m.height) {
                col = 0;
            }
            stack_.clear();
        }

        struct data_ {
            bb::Bitboard board[2];
            int counter;
            int height[NUM_COLS];
        } m;

        std::vector<data_> stack_;
    };

}
