//
// Created by Yngvi Björnsson on 11.5.2022.
//

#ifndef ABG8X8_MOVE_H
#define ABG8X8_MOVE_H

#include "../base/types.h"
#include "../utils/utils.h"

namespace game {

    using move_iterator = utils::void_ptr_iterator;

    using move_id = uint64_t;

    class IMovelist {
    public:
        [[nodiscard]] virtual bool empty() const = 0;
        [[nodiscard]] virtual int get_size() const = 0;
        virtual void place_first(move_id) = 0;
        virtual bool is_move_none(move_iterator) = 0;
        virtual move_iterator begin() = 0;
        virtual move_iterator end() = 0;

        [[nodiscard]] virtual move_id as_move_id(move_iterator) const = 0;

        virtual ~IMovelist() = default;
    };

    using MovelistPtr = std::unique_ptr<IMovelist, std::function<void(void *)>>;

    template<class T, int SIZE>
    class Movelist : public IMovelist {
    public:
        Movelist() : len_(0) {}

        [[nodiscard]] bool is_move_none(move_iterator mi) override { return *(mi.as<T>()) == this->no_move; }
        [[nodiscard]] bool empty() const override { return len_ == 0; }
        [[nodiscard]] int get_size() const override { return len_; }
        void place_first(move_id id) override {
            T move(id);
            for (unsigned int i=0 ; i < len_; ++i) {
                if (move == moves_[i]) {
                    std::swap(moves_[0], moves_[i]);
                    break;
                }
            }
        }
        [[nodiscard]] game::move_iterator begin() override { return {&(moves_[0]), sizeof(T)}; }
        [[nodiscard]] game::move_iterator end() override { return {&(moves_[len_]), sizeof(T)}; }
        [[nodiscard]] move_id as_move_id(move_iterator mi) const override {
            return static_cast<move_id>(*(mi.as<T>()));
        }

        // Others.
        [[nodiscard]] T get_move(int n) const { return moves_[n]; }

        bool add(const T& move) {
            if (len_ < SIZE) {
                moves_[len_++] = move;
                return true;
            }
            return false;
        }

    protected:
        unsigned int len_;
        T moves_[SIZE];
        static inline T no_move;
    };

    template<typename T, int SIZE>
    class MovelistPooled : public Movelist<T,SIZE> {
    public:

        template<typename ... Args>
        static auto new_pooled(Args&& ... args) {
            return pool.make_managed(std::forward<Args>(args) ...);
        }

        static void clear_pooled() {
            pool.clear();
        }

    private:
        static inline utils::ObjectPool<MovelistPooled<T,SIZE>> pool;
    };

}



namespace mm {

    // NOTE: should be in types?
    struct Move {
    public:
        Move() = default;
        explicit Move(game::move_id mid) : id(mid) {}
        Move(Square from, Square to, bool capture = false) :
                m({(uint16_t) from, (uint16_t) to, (uint16_t) capture}) {}  //NOTE casting from square
        [[nodiscard]] inline Square from() const { return (Square) m.from; }
        [[nodiscard]] inline Square to() const { return (Square) m.to; }
        [[nodiscard]] inline bool is_capture() const { return (bool) m.capture; }

        [[nodiscard]] explicit operator game::move_id() const { return id; }

        bool operator==(const Move& rhs) const {
            return (m.from == rhs.m.from) && (m.to == rhs.m.to) && (m.capture == rhs.m.capture);
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
        ss << move.from() << (move.is_capture() ? 'x' : '-') << move.to();
        return ss.str();
    }

    inline std::ostream& operator<<(std::ostream& os, const Move& move) {
        os << to_string(move);
        return os;
    }

    inline std::istream& operator>>(std::istream& is, Move& move) {
        Square from, to;
        char c ='\0';
        is >> from >> c >> to;
        switch ( c ) {
            case '-' :  move = Move(from, to); break;
            case 'x' :  move = Move(from, to, true); break;
            default:    is.setstate(std::ios_base::failbit); break;
        }
        return is;
    }

    inline std::string str(const Move& move) {
        std::stringstream ss;
        ss << move;
        return ss.str();
    }

}
using namespace mm;

#endif //ABG8X8_MOVE_H
