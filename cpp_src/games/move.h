#pragma once
#include "../base/types.h"
#include "../utils/utils.h"

namespace game {
    using move_id = uint64_t;
}



namespace mm {

    // NOTE: should be in types?
    struct Move {
    public:
        Move() = default;
        explicit Move(game::move_id mid) : id(mid) {

        }

        Move(
            Square from, 
            Square to, 
            bool capture = false
        ) : m({
            (uint16_t) from, 
            (uint16_t) to, 
            (uint16_t) capture
        }) {}  //NOTE casting from square

        [[nodiscard]] inline Square from() const { 
            return (Square) m.from; 
        }

        [[nodiscard]] inline Square to() const { 
            return (Square) m.to; 
        }

        [[nodiscard]] inline bool is_capture() const { 
            return (bool) m.capture; 
        }

        [[nodiscard]] explicit operator game::move_id() const { 
            return   m.from | (m.to << 6) | (m.capture << 12)  ;
        }

        bool operator==(const Move& rhs) const {
            return (m.from == rhs.m.from) 
                && (m.to == rhs.m.to) 
                && (m.capture == rhs.m.capture);
        }

    private:
        struct data_ {  // NOTE: compile-time assert.
            uint16_t from    : 6;
            uint16_t to      : 6;
            uint16_t capture : 1;
        };
        union {
            data_ m;
            uint64_t id;
        };
    };

    const Move no_move = Move(ss::Square::a1, ss::Square::a1, false);

    inline std::string to_string(const Move& move) {

        std::stringstream ss;

        ss << move.from() 
           << (move.is_capture() ? 'x' : '-') 
           << move.to();

        return ss.str();
    }

    inline std::ostream& operator<<(
        std::ostream& os, 
        const Move& move
    ) {
        os << to_string(move);
        return os;
    }

    // inline std::istream& operator>>(
    //     std::istream& is, 
    //     Move& move
    // ) {
    //     Square from, to;
    //     char c ='\0';

    //     is >> from >> c >> to;

    //     switch ( c ) {
    //         case '-' :  move = Move(from, to); break;
    //         case 'x' :  move = Move(from, to, true); break;
    //         default:    is.setstate(std::ios_base::failbit); break;
    //     }
    //     return is;
    // }

    inline game::move_id from_str(const std::string& str) {

        char from_file, from_rank, mid, to_file, to_rank;
        std::stringstream ss(str);
        ss >> from_file >> from_rank >> mid >> to_file >> to_rank;

        int from = (from_file - 'a') + (from_rank - '1') * 8;
        int to = (to_file - 'a') + (to_rank - '1') * 8;
        int cap = (mid == 'x') ? 1 : 0;

        return from | (to << 6) | (cap << 12);

    }

    inline std::string str(const Move& move) {
        std::stringstream ss;
        ss << move;
        return ss.str();
    }

}
using namespace mm;

