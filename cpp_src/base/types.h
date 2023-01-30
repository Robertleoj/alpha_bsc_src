//
// Created by Yngvi Björnsson on 7.5.2022.
//

#ifndef ABG8X8_TYPES_H
#define ABG8X8_TYPES_H

#include <iostream>
#include <sstream>
#include <string>
#include <limits>
#include <assert.h>

constexpr int FILES = 8;
constexpr int RANKS = 8;

constexpr int PLAY_FILES = FILES;
constexpr int PLAY_RANKS = RANKS;

constexpr bool BOARD_IS_8x8 = FILES==8 && RANKS==8 && FILES==PLAY_FILES && RANKS==PLAY_RANKS;

namespace ss {

    static_assert(FILES == 8, "Invalid number of <files> specified.");
    static_assert(RANKS == 8, "Invalid number of <ranks> specified.");
    static_assert(PLAY_FILES >= 1 && PLAY_FILES <= FILES, "Invalid number of <play_files> specified.");
    static_assert(PLAY_RANKS >= 1 && PLAY_RANKS <= RANKS, "Invalid number of <play_ranks> specified.");

    enum class Square {
        a1, b1, c1, d1, e1, f1, g1, h1,
        a2, b2, c2, d2, e2, f2, g2, h2,
        a3, b3, c3, d3, e3, f3, g3, h3,
        a4, b4, c4, d4, e4, f4, g4, h4,
        a5, b5, c5, d5, e5, f5, g5, h5,
        a6, b6, c6, d6, e6, f6, g6, h6,
        a7, b7, c7, d7, e7, f7, g7, h7,
        a8, b8, c8, d8, e8, f8, g8, h8 };

    inline Square operator++(Square& sq) { sq = Square((int)sq + 1); return sq; }
    inline Square operator--(Square& sq) { sq = Square((int)sq - 1); return sq; }

    /*
    constexpr std::initializer_list<Square> Squares = {
        Square::a1, Square::b1, Square::c1, Square::d1, Square::e1, Square::f1, Square::g1, Square::h1,
        Square::a2, Square::b2, Square::c2, Square::d2, Square::e2, Square::f2, Square::g2, Square::h2,
        Square::a3, Square::b3, Square::c3, Square::d3, Square::e3, Square::f3, Square::g3, Square::h3,
        Square::a4, Square::b4, Square::c4, Square::d4, Square::e4, Square::f4, Square::g4, Square::h4,
        Square::a5, Square::b5, Square::c5, Square::d5, Square::e5, Square::f5, Square::g5, Square::h5,
        Square::a6, Square::b6, Square::c6, Square::d6, Square::e6, Square::f6, Square::g6, Square::h6,
        Square::a7, Square::b7, Square::c7, Square::d7, Square::e7, Square::f7, Square::g7, Square::h7,
        Square::a8, Square::b8, Square::c8, Square::d8, Square::e8, Square::f8, Square::g8, Square::h8 };
    */

    enum class File {
        file_a, file_b, file_c, file_d, file_e, file_f, file_g, file_h
    };

    constexpr static File file_end = (File)(PLAY_FILES-1);

    inline File operator++(File& file) { file = File((int)file + 1); return file; }
    inline File operator--(File& file) { file = File((int)file - 1); return file; }

    /*
    constexpr std::initializer_list<File> Files = {
        File::file_a, File::file_b, File::file_c, File::file_d, File::file_e, File::file_f, File::file_g, File::file_h
    };
    */

    enum class Rank {
        rank_1, rank_2, rank_3, rank_4, rank_5, rank_6, rank_7, rank_8
    };

    constexpr static Rank rank_end = (Rank)(PLAY_RANKS-1);

    inline Rank operator++(Rank& rank) { rank = Rank((int)rank + 1); return rank; }
    inline Rank operator--(Rank& rank) { rank = Rank((int)rank - 1); return rank; }

    /*
    constexpr std::initializer_list<Rank> Ranks = {
        Rank::rank_1, Rank::rank_2, Rank::rank_3, Rank::rank_4, Rank::rank_5, Rank::rank_6, Rank::rank_7, Rank::rank_8
    };
    */

    inline Square square(File file, Rank rank) {
        return Square(FILES * (int)rank + (int)file);
    }

    inline File file(Square sq) {
        return (File)( (int)sq % FILES);
    }

    inline Rank rank(Square sq) {
        return (Rank)((int)sq / FILES);
    }

    enum Direction : int {
        dN  =  FILES,
        dE  =  1,
        dS  = -dN,
        dW  = -dE,
        dNE = dN + dE,
        dSE = dS + dE,
        dSW = dS + dW,
        dNW = dN + dW
    };
    
    inline std::ostream& operator<<(std::ostream& os, const File& file) {
        assert(FILES <= 8);
        os << "abcdefgh"[(int)file];
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const Rank& rank) {
        assert(RANKS <= 8);
        os << "12345678"[(int)rank];
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const Square& square) {
        os << file(square) << rank(square);
        return os;
    }

    inline std::istream& operator>>(std::istream& is, Square& sq) {
        char file, rank;
        is >> file >> rank;
        if (file>='a' && file<='h' && rank>=1 && rank<=8) {
            sq = square((File)(file-'a'), (Rank)(rank-'0'));
        }
        else { is.setstate(std::ios_base::failbit); }
        return is;
    }

}
using namespace ss;


namespace pp {

    enum Player { First=0, Second=1 };

    inline Player other(Player pl) {
        return (pl == First) ? Second : First;
    }

}
using namespace pp;


namespace out {

    enum class Outcome {
        Loss = 0, Tie = 1, Win = 2, Undecided = 3
    };

    static inline std::string str(Outcome outcome) {
        static std::string text[] = { "0-1", "1/2-1/2", "1-0", "*"};
        return text[(int)outcome];
    }

}
using namespace out;


namespace vv {

    enum class Value : int {};
    constexpr Value operator-(Value v) { return (Value) -(int)v; }
    constexpr Value operator-(Value v, int n) { return (Value) ((int)v - n); }
    constexpr Value operator+(Value v, int n) { return (Value) ((int)v + n); }
    /*
    constexpr bool operator>(Value lhs, Value rhs) { return (int) lhs > (int) rhs; }
    constexpr bool operator>=(Value lhs, Value rhs) { return (int) lhs >= (int) rhs; }
    constexpr bool operator<(Value lhs, Value rhs) { return (int) lhs < (int) rhs; }
    constexpr bool operator<=(Value lhs, Value rhs) { return (int) lhs <= (int) rhs; }
    constexpr bool operator==(Value lhs, Value rhs) { return (int) lhs == (int) rhs; }
    constexpr bool operator!=(Value lhs, Value rhs) { return (int) lhs != (int) rhs; }
*/
    //using Value = int;

    constexpr Value infinity = (Value)std::numeric_limits<int>::max();
    constexpr Value win = (Value)10000;
    constexpr Value loss = (Value) -win;
    constexpr Value draw = (Value)0;

    static inline bool is_win(Value value) {
        return (value >= (win - 999));
    }

    static inline bool is_loss(Value value) {
        return is_win(-value);
    }

    static inline bool is_draw(Value value) {
        return (value == draw);
    }

    static inline bool is_terminal(Value value) {
        return is_win(value) || is_loss(value) || is_draw(value);
    }

    static inline bool is_win_in(int n, Value value) {
        return (value == win - n);
    }

    static inline bool is_loss_in(int n, Value value) {
        return (value == loss + n);
    }

    static inline bool is_win_or_loss_in(int n, Value value) {
        return is_win_in(n, value) || is_loss_in(n, value);
    }

    static inline Value value_of(out::Outcome outcome) {
        static Value val[] = {loss, draw, win, (Value)1};
        return val[(int)outcome];
    }

    enum Type {
        Exact = 0, LowerBound = 1, UpperBound = 2
    };

}
using namespace vv;

namespace state {

    struct state_t {
        char board[FILES][RANKS] = { { 0 } };
        char side_to_move = 0;
    };

    inline void init(struct state_t& state) {
        for (auto rank=Rank::rank_1; rank <= rank_end; ++rank) {
            for (auto file=File::file_a; file <= file_end; ++file) {
                state.board[(int)file][(int)rank] = '.';
            }
        }
        state.side_to_move = '.';
    }

}
using namespace state;

#endif //ABG8X8_TYPES_H
