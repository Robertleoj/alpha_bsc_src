//
// Created by Yngvi Björnsson on 7.5.2022.
//

#include "breakthrough.h"
#include <cassert>
#include <sstream>
#include <assert.h>

constexpr static bb::Bitboard bb_file_end = bb::bb_file[(int)file_end];
constexpr static bb::Bitboard bb_rank_end = bb::bb_rank[(int)rank_end];
using namespace games;

Breakthrough::Breakthrough()
   : m{ First, {bb::empty, bb::empty}}
{
    init();
}

Breakthrough::~Breakthrough()
{
    Movelist::clear_pooled();
}


void Breakthrough::setup()
{
    init();
    assert(is_legal());
}


bool Breakthrough::setup(const struct state_t& state)
{
    m.to_move = First;
    m.bb_pieces[First] = m.bb_pieces[Second] = bb::empty;
    for (auto rank= Rank::rank_1; rank<=rank_end; ++rank) {
        for (auto file= File::file_a; file<=file_end; ++file) {
            switch(state.board[(int)file][(int)rank]) {
                case 'w': m.bb_pieces[First] |= bb::square(square(file, rank)); break;
                case 'b': m.bb_pieces[Second] |= bb::square(square(file, rank)); break;
                case '.': break;
                default: return false;
            }
        }
    }
    switch(state.side_to_move) {
        case 'w': m.to_move = First; break;
        case 'b': m.to_move = Second; break;
        default: return false;
    }
    return is_legal();
}


bool Breakthrough::setup(const std::string& fen)
{
    std::stringstream ss(fen);
    std::string board, turn;
    ss >> board >> turn;
    if (board.empty() || turn.empty()) { return false; }

    struct state_t state;
    state::init(state);
    int i = 0;
    for (auto rank=rank_end; rank >= Rank::rank_1; --rank) {
        for (auto file=File::file_a; file<=file_end; ++file) {
            if (board[i] == '/') { break; }
            state.board[(int)file][(int)rank] = board[i];
            if (++i >= (int)board.size()) { break; }
        }
        if (board[i] == '/') {
            if (++i >= (int)board.size()) { break; }
        }
    }
    state.side_to_move = turn[0];
    return setup(state);
}


void Breakthrough::get_state(state_t& state) const
{
    state::init(state);
    bb::Bitboard bb = m.bb_pieces[First];
    while (bb) {
        Square sq = bb::pop_lsb(bb);
        state.board[(int)file(sq)][(int)rank(sq)] = 'w';
    }
    bb = m.bb_pieces[Second];
    while (bb) {
        Square sq = bb::pop_lsb(bb);
        state.board[(int)file(sq)][(int)rank(sq)] = 'b';
    }
    state.side_to_move = (m.to_move == First) ? 'w' : 'b';
}


std::string Breakthrough::to_string() const {
    std::stringstream ss;
    display(ss, "/");
    return ss.str();
}


void Breakthrough::display(std::ostream& os, const std::string& delimiter) const {
    for (auto rank=rank_end; rank>=Rank::rank_1; --rank) {
        if (rank != rank_end) {
            os << delimiter;
        }
        for (auto file=File::file_a; file<=file_end; ++file) {
            Square sq = ss::square(file, rank);
            if      (m.bb_pieces[First] & bb::square(sq))  { os << 'w'; }
            else if (m.bb_pieces[Second] & bb::square(sq)) { os << 'b'; }
            else                                          { os << '.'; }
        }
    }
    os << ' ' << ((m.to_move==First) ? 'w' : 'b');
}


void Breakthrough::make(const Move& move) {
    bb::Bitboard  bb_sq_to = bb::square(move.to());
    m.bb_pieces[m.to_move] ^= bb::square(move.from()) | bb_sq_to;
    m.to_move = other(m.to_move);
    m.bb_pieces[m.to_move] ^= bb_sq_to & bb::set_if[move.is_capture()];
}


void Breakthrough::retract(const Move& move) {
    bb::Bitboard  bb_sq_to = bb::square(move.to());
    m.bb_pieces[m.to_move] ^= bb_sq_to & bb::set_if[move.is_capture()];
    m.to_move = other(m.to_move);
    m.bb_pieces[m.to_move] ^= bb::square(move.from()) | bb_sq_to;
}


void Breakthrough::push() {
    stack_.push_back(m);
}


void Breakthrough::pop() {
    if (!stack_.empty()) {
        m = stack_.back();
        stack_.pop_back();
    }
}


bool Breakthrough::is_key_unique() const {
    // True for legal positions.
    return true;
}


game::key Breakthrough::get_key() const  // NOTE. check - correctness
{
    game::key k;
    if (get_to_move() == First) {
        k = m.bb_pieces[0] | bb::bb_rank[(int)Rank::rank_8];  // NOTE: FILES or PLAY_FILES.
        k = (k << 64) | m.bb_pieces[1];
    }
    else {
        k = m.bb_pieces[1] | bb::bb_rank[(int)Rank::rank_1];
        k = (k << 64) | m.bb_pieces[0];
    }
    return k;
}


Outcome Breakthrough::outcome(Player pl) const {
    if (pl == First) {
        if      (is_terminal<First>()) { return Outcome::Loss; }
        else if (is_terminal<Second>()) {  return Outcome::Win; }
    }
    else {
        if      (is_terminal<Second>()) { return Outcome::Loss; }
        else if (is_terminal<First>()) { return Outcome::Win; }
    }
    return Outcome::Undecided;
}


Outcome Breakthrough::outcome() const {
    return outcome(m.to_move);
}


int Breakthrough::generate(Movelist& move_list) const {
    return m.to_move == First ? generate<First>(move_list) : generate<Second>(move_list);
}


bool Breakthrough::is_terminal(const Move& move) const
{
    return m.to_move == First ? is_terminal<First>(move) : is_terminal<Second>(move);
}


bool Breakthrough::is_terminal() const
{
    return m.to_move == First ? is_terminal<First>() : is_terminal<Second>();
}


bool Breakthrough::is_legal() const
{
    return !is_terminal<First>() || !is_terminal<Second>();
}


bool Breakthrough::is_legal(const std::string& move_str, mm::Move& move) const
{
    Movelist move_list;
    generate(move_list);
    for (auto i = 0, len = move_list.get_size(); i < len; ++i) {
        if (str(move_list.get_move(i)) == move_str) {
            move = move_list.get_move(i);
            return true;
        }
    }
    return false;
}


int Breakthrough::get_piece_count(Player player) const
{
    return player == First ? get_piece_count<First>() : get_piece_count<Second>();
}


Player Breakthrough::get_to_move() const
{
    return m.to_move;
}


template<Player pl, typename T>
static constexpr T set_side(T w, T b)
{
    return (pl == First) ? w : b;
}


void Breakthrough::init( )
{
    m.to_move = First;
    m.bb_pieces[First] = m.bb_pieces[Second] = bb::empty;
    for (auto col = File::file_a; col <= file_end; ++col) {
        m.bb_pieces[First] |= bb::square(square(col, Rank::rank_1));
        m.bb_pieces[First] |= bb::square(square(col, Rank::rank_2));
        m.bb_pieces[Second] |= bb::square(square(col, Rank((int)rank_end-1)));
        m.bb_pieces[Second] |= bb::square(square(col, rank_end));
    }
}


Board Breakthrough::get_board() const
{
    Board board;
    board.to_move = get_to_move();
    board.bbs.push_back(m.bb_pieces[0]);
    board.bbs.push_back(m.bb_pieces[1]);
    return board;
}


bool Breakthrough::set(const std::string& )  {
    return false;  // TBA
}


template<Player Pl>
int Breakthrough::generate(Movelist& move_list) const
{
    constexpr Player player = set_side<Pl>(First, Second);
    constexpr Player other = set_side<Pl>(Second, First);
    constexpr Direction Forward = set_side<Pl>(dN, dS);
    constexpr Direction ForwardLeft = set_side<Pl>(dNW, dSE);
    constexpr Direction ForwardRight = set_side<Pl>(dNE, dSW);
    constexpr bb::Bitboard bb_LeftmostFile = set_side<Pl>(bb::bb_a, bb_file_end);
    constexpr bb::Bitboard bb_RightmostFile = set_side<Pl>(bb_file_end, bb::bb_a);

    // Captures.
    bb::Bitboard bb_pieces_except_leftmost_file = (m.bb_pieces[player] & ~bb_LeftmostFile);
    bb::Bitboard bb_pieces_except_rightmost_file = (m.bb_pieces[player] & ~bb_RightmostFile);
    bb::Bitboard bb = bb::shift(bb_pieces_except_leftmost_file, ForwardLeft) & m.bb_pieces[other];
    while (bb) {
        Square sq_to = bb::pop_lsb(bb);
        move_list.add(Move(Square((int)sq_to - ForwardLeft), sq_to, true));
    }
    bb = bb::shift(bb_pieces_except_rightmost_file, ForwardRight) & m.bb_pieces[other];
    while (bb) {
        Square sq_to = bb::pop_lsb(bb);
        move_list.add(Move(Square((int)sq_to - ForwardRight), sq_to, true));
    }

    // Non-captures
    bb::Bitboard bb_empty_sq = ~(m.bb_pieces[First] | m.bb_pieces[Second]);
    bb = bb::shift(bb_pieces_except_leftmost_file, ForwardLeft) & bb_empty_sq;
    while (bb) {
        Square sq_to = bb::pop_lsb(bb);
        move_list.add(Move(Square((int)sq_to - ForwardLeft), sq_to));
    }
    bb = bb::shift(m.bb_pieces[player], Forward) & bb_empty_sq;
    while (bb) {
        Square sq_to = bb::pop_lsb(bb);
        move_list.add(Move(Square((int)sq_to - Forward), sq_to));
    }
    bb = bb::shift(bb_pieces_except_rightmost_file, ForwardRight) & bb_empty_sq;
    while (bb) {
        Square sq_to = bb::pop_lsb(bb);
        move_list.add(Move(Square((int)sq_to - ForwardRight), sq_to));
    }

    return move_list.get_size();
}


template<Player Pl>
bool Breakthrough::is_terminal() const
{
    constexpr Player other = set_side<Pl>(Second, First);
    constexpr bb::Bitboard bb_home_rank = set_side<Pl>(bb::bb_1, bb_rank_end);
    return ((m.bb_pieces[other] & bb_home_rank) != 0ULL) || (bb::pop_count(m.bb_pieces[Pl]) == 0);
}


template<Player Pl>
bool Breakthrough::is_terminal(const Move& move) const
{
    constexpr Player other = set_side<Pl>(Second, First);
    constexpr Rank back_rank = set_side<Pl>(rank_end, Rank::rank_1);
    return (rank(move.to()) == back_rank) ||
           (move.is_capture() && bb::pop_count(m.bb_pieces[other]) == 1);
}


template<Player Pl>
int Breakthrough::get_piece_count() const {
    return bb::pop_count(m.bb_pieces[Pl]);
}
