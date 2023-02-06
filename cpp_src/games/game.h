//
// Created by Yngvi Björnsson on 7.5.2022.
//
#pragma once

#include <memory>
#include <functional>
#include <ostream>
#include <string>
#include "move.h"
#include "../base/types.h"
#include "../base/bitboard.h"
#include "../base/board.h"


namespace game {

    using key = __int128;  // NOTE: none standard

    // IGameSearchable
    class IGameSearchable {
    public:
        virtual  MovelistPtr moves() = 0;

        virtual Board get_board() const = 0;

        virtual bool make(move_iterator mi) = 0;
        virtual void retract(move_iterator mi) = 0;
        virtual void push() = 0;
        virtual void pop() = 0;

        [[nodiscard]] virtual bool is_terminal() const = 0;
        [[nodiscard]] virtual Outcome outcome() const = 0;
        [[nodiscard]] virtual Outcome outcome(Player pl) const = 0;

        [[nodiscard]] virtual bool is_key_unique() const = 0;
        [[nodiscard]] virtual key get_key() const = 0;

        virtual ~IGameSearchable() = default;
    };


    class IGame : public IGameSearchable {
    public:
        static constexpr char* default_delimiter = (char*)"\n";

        // Setup methods.
        [[maybe_unused]] virtual void setup() = 0;

        // Accessor methods.
        [[nodiscard]] virtual Player get_to_move() const = 0;

        // Other methods.
        virtual void display(
            std::ostream& os, 
            const std::string& delimiter = default_delimiter
        ) const = 0;

        [[nodiscard]] virtual std::string move_as_str(move_iterator it) const = 0;
        [[nodiscard]] virtual std::string move_as_str(move_id id) const = 0;

        virtual bool set(const std::string& pos) = 0;

        // Destructor.
        ~IGame() override = default;
    };


    inline bool make_move_if_legal(
        IGame& game, 
        const std::string& move_str
    ) {
        auto moves = game.moves();
        for(auto it = moves->begin(); it != moves->end(); ++it) {

            if(game.move_as_str(it) == move_str) {
                game.make(it);
                return true;
            }
        }
        return false;
    }

}

