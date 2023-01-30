//
// Created by Yngvi Björnsson on 7.5.2022.
//

#ifndef ABG8X8_BITBOARD_H
#define ABG8X8_BITBOARD_H

#include "types.h"
#include <assert.h>

namespace bb {

    using Bitboard = uint64_t;
/*
    static std::string to_string(const Bitboard& bb) {
        return std::bitset<64>(bb).to_string();
    }
*/
    constexpr Bitboard empty = 0b0000000000000000000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard full  = 0b1111111111111111111111111111111111111111111111111111111111111111ULL;
    constexpr Bitboard set_if[] = {empty, full};  // NOTE NAMING.

    constexpr Bitboard bb_a1 = 0b0000000000000000000000000000000000000000000000000000000000000001ULL;
    constexpr Bitboard bb_b1 = 0b0000000000000000000000000000000000000000000000000000000000000010ULL;
    constexpr Bitboard bb_c1 = 0b0000000000000000000000000000000000000000000000000000000000000100ULL;
    constexpr Bitboard bb_d1 = 0b0000000000000000000000000000000000000000000000000000000000001000ULL;
    constexpr Bitboard bb_e1 = 0b0000000000000000000000000000000000000000000000000000000000010000ULL;
    constexpr Bitboard bb_f1 = 0b0000000000000000000000000000000000000000000000000000000000100000ULL;
    constexpr Bitboard bb_g1 = 0b0000000000000000000000000000000000000000000000000000000001000000ULL;
    constexpr Bitboard bb_h1 = 0b0000000000000000000000000000000000000000000000000000000010000000ULL;
    constexpr Bitboard bb_a2 = 0b0000000000000000000000000000000000000000000000000000000100000000ULL;
    constexpr Bitboard bb_b2 = 0b0000000000000000000000000000000000000000000000000000001000000000ULL;
    constexpr Bitboard bb_c2 = 0b0000000000000000000000000000000000000000000000000000010000000000ULL;
    constexpr Bitboard bb_d2 = 0b0000000000000000000000000000000000000000000000000000100000000000ULL;
    constexpr Bitboard bb_e2 = 0b0000000000000000000000000000000000000000000000000001000000000000ULL;
    constexpr Bitboard bb_f2 = 0b0000000000000000000000000000000000000000000000000010000000000000ULL;
    constexpr Bitboard bb_g2 = 0b0000000000000000000000000000000000000000000000000100000000000000ULL;
    constexpr Bitboard bb_h2 = 0b0000000000000000000000000000000000000000000000001000000000000000ULL;
    constexpr Bitboard bb_a3 = 0b0000000000000000000000000000000000000000000000010000000000000000ULL;
    constexpr Bitboard bb_b3 = 0b0000000000000000000000000000000000000000000000100000000000000000ULL;
    constexpr Bitboard bb_c3 = 0b0000000000000000000000000000000000000000000001000000000000000000ULL;
    constexpr Bitboard bb_d3 = 0b0000000000000000000000000000000000000000000010000000000000000000ULL;
    constexpr Bitboard bb_e3 = 0b0000000000000000000000000000000000000000000100000000000000000000ULL;
    constexpr Bitboard bb_f3 = 0b0000000000000000000000000000000000000000001000000000000000000000ULL;
    constexpr Bitboard bb_g3 = 0b0000000000000000000000000000000000000000010000000000000000000000ULL;
    constexpr Bitboard bb_h3 = 0b0000000000000000000000000000000000000000100000000000000000000000ULL;
    constexpr Bitboard bb_a4 = 0b0000000000000000000000000000000000000001000000000000000000000000ULL;
    constexpr Bitboard bb_b4 = 0b0000000000000000000000000000000000000010000000000000000000000000ULL;
    constexpr Bitboard bb_c4 = 0b0000000000000000000000000000000000000100000000000000000000000000ULL;
    constexpr Bitboard bb_d4 = 0b0000000000000000000000000000000000001000000000000000000000000000ULL;
    constexpr Bitboard bb_e4 = 0b0000000000000000000000000000000000010000000000000000000000000000ULL;
    constexpr Bitboard bb_f4 = 0b0000000000000000000000000000000000100000000000000000000000000000ULL;
    constexpr Bitboard bb_g4 = 0b0000000000000000000000000000000001000000000000000000000000000000ULL;
    constexpr Bitboard bb_h4 = 0b0000000000000000000000000000000010000000000000000000000000000000ULL;
    constexpr Bitboard bb_a5 = 0b0000000000000000000000000000000100000000000000000000000000000000ULL;
    constexpr Bitboard bb_b5 = 0b0000000000000000000000000000001000000000000000000000000000000000ULL;
    constexpr Bitboard bb_c5 = 0b0000000000000000000000000000010000000000000000000000000000000000ULL;
    constexpr Bitboard bb_d5 = 0b0000000000000000000000000000100000000000000000000000000000000000ULL;
    constexpr Bitboard bb_e5 = 0b0000000000000000000000000001000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_f5 = 0b0000000000000000000000000010000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_g5 = 0b0000000000000000000000000100000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_h5 = 0b0000000000000000000000001000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_a6 = 0b0000000000000000000000010000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_b6 = 0b0000000000000000000000100000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_c6 = 0b0000000000000000000001000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_d6 = 0b0000000000000000000010000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_e6 = 0b0000000000000000000100000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_f6 = 0b0000000000000000001000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_g6 = 0b0000000000000000010000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_h6 = 0b0000000000000000100000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_a7 = 0b0000000000000001000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_b7 = 0b0000000000000010000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_c7 = 0b0000000000000100000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_d7 = 0b0000000000001000000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_e7 = 0b0000000000010000000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_f7 = 0b0000000000100000000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_g7 = 0b0000000001000000000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_h7 = 0b0000000010000000000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_a8 = 0b0000000100000000000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_b8 = 0b0000001000000000000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_c8 = 0b0000010000000000000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_d8 = 0b0000100000000000000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_e8 = 0b0001000000000000000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_f8 = 0b0010000000000000000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_g8 = 0b0100000000000000000000000000000000000000000000000000000000000000ULL;
    constexpr Bitboard bb_h8 = 0b1000000000000000000000000000000000000000000000000000000000000000ULL;

    constexpr Bitboard bb_a = bb_a1 | bb_a2 | bb_a3 | bb_a4 | bb_a5 | bb_a6 | bb_a7 | bb_a8;
    constexpr Bitboard bb_b = bb_b1 | bb_b2 | bb_b3 | bb_b4 | bb_b5 | bb_b6 | bb_b7 | bb_b8;
    constexpr Bitboard bb_c = bb_c1 | bb_c2 | bb_c3 | bb_c4 | bb_c5 | bb_c6 | bb_c7 | bb_c8;
    constexpr Bitboard bb_d = bb_d1 | bb_d2 | bb_d3 | bb_d4 | bb_d5 | bb_d6 | bb_d7 | bb_d8;
    constexpr Bitboard bb_e = bb_e1 | bb_e2 | bb_e3 | bb_e4 | bb_e5 | bb_e6 | bb_e7 | bb_e8;
    constexpr Bitboard bb_f = bb_f1 | bb_f2 | bb_f3 | bb_f4 | bb_f5 | bb_f6 | bb_f7 | bb_f8;
    constexpr Bitboard bb_g = bb_g1 | bb_g2 | bb_g3 | bb_g4 | bb_g5 | bb_g6 | bb_g7 | bb_g8;
    constexpr Bitboard bb_h = bb_h1 | bb_h2 | bb_h3 | bb_h4 | bb_h5 | bb_h6 | bb_h7 | bb_h8;

    constexpr Bitboard bb_1 = bb_a1 | bb_b1 | bb_c1 | bb_d1 | bb_e1 | bb_f1 | bb_g1 | bb_h1;
    constexpr Bitboard bb_2 = bb_a2 | bb_b2 | bb_c2 | bb_d2 | bb_e2 | bb_f2 | bb_g2 | bb_h2;
    constexpr Bitboard bb_3 = bb_a3 | bb_b3 | bb_c3 | bb_d3 | bb_e3 | bb_f3 | bb_g3 | bb_h3;
    constexpr Bitboard bb_4 = bb_a4 | bb_b4 | bb_c4 | bb_d4 | bb_e4 | bb_f4 | bb_g4 | bb_h4;
    constexpr Bitboard bb_5 = bb_a5 | bb_b5 | bb_c5 | bb_d5 | bb_e5 | bb_f5 | bb_g5 | bb_h5;
    constexpr Bitboard bb_6 = bb_a6 | bb_b6 | bb_c6 | bb_d6 | bb_e6 | bb_f6 | bb_g6 | bb_h6;
    constexpr Bitboard bb_7 = bb_a7 | bb_b7 | bb_c7 | bb_d7 | bb_e7 | bb_f7 | bb_g7 | bb_h7;
    constexpr Bitboard bb_8 = bb_a8 | bb_b8 | bb_c8 | bb_d8 | bb_e8 | bb_f8 | bb_g8 | bb_h8;

    static constexpr Bitboard bb_sq[] = {
            bb_a1, bb_b1, bb_c1, bb_d1, bb_e1, bb_f1, bb_g1, bb_h1,
            bb_a2, bb_b2, bb_c2, bb_d2, bb_e2, bb_f2, bb_g2, bb_h2,
            bb_a3, bb_b3, bb_c3, bb_d3, bb_e3, bb_f3, bb_g3, bb_h3,
            bb_a4, bb_b4, bb_c4, bb_d4, bb_e4, bb_f4, bb_g4, bb_h4,
            bb_a5, bb_b5, bb_c5, bb_d5, bb_e5, bb_f5, bb_g5, bb_h5,
            bb_a6, bb_b6, bb_c6, bb_d6, bb_e6, bb_f6, bb_g6, bb_h6,
            bb_a7, bb_b7, bb_c7, bb_d7, bb_e7, bb_f7, bb_g7, bb_h7,
            bb_a8, bb_b8, bb_c8, bb_d8, bb_e8, bb_f8, bb_g8, bb_h8
    };

    static constexpr Bitboard bb_file[] = {
            bb_a, bb_b, bb_c, bb_d, bb_e, bb_f, bb_g, bb_h
    };

    static constexpr Bitboard bb_rank[] = {
            bb_1, bb_2, bb_3, bb_4, bb_5, bb_6, bb_7, bb_8
    };

    constexpr Bitboard square(ss::Square sq) {
        return bb_sq[(int)sq];
    }

    constexpr Bitboard file(File file) {
        return bb_file[(int)file];
    }

    constexpr Bitboard rank(Rank rank) {
        return bb_rank[(int)rank];
    }

    inline ss::Square lsb(Bitboard b) {
        assert(b);
        return ss::Square(__builtin_ctzll(b));
    }

    inline ss::Square pop_lsb(Bitboard &b) {
        assert(b);
        const ss::Square s = lsb(b);
        b &= b - 1;
        return s;
    }

    inline int pop_count(Bitboard b) {
        return __builtin_popcountll(b);
    }

    inline Bitboard shift(Bitboard bb, int n) {
        return (n >= 0) ? (bb << (unsigned int) n) : (bb >> (unsigned int) (-n));
    }


//#define USE_BUILTIN_MACROS
#ifdef USE_BUILTIN_MACROS
    inline Bitboard reverse(Bitboard bb) {
        assert(FILES == 8  && RANKS == 8);
        assert(PLAY_FILES == FILES && PLAY_RANKS == RANKS);
        return __builtin_bitreverse64(bb);
    }
#else

    template<typename T, T m, uint k>
    static inline T swap_bits(T p) {
        T q = ((p >> k) ^ p) & m;
        return p ^ q ^ (q << k);
    }

    inline Bitboard reverse(Bitboard bb) {
        assert(BOARD_IS_8x8);
        static constexpr uint64_t m0 = 0x5555555555555555LLU;
        static constexpr uint64_t m1 = 0x0300c0303030c303LLU;
        static constexpr uint64_t m2 = 0x00c0300c03f0003fLLU;
        static constexpr uint64_t m3 = 0x00000ffc00003fffLLU;
        bb = ((bb >> 1u) & m0) | (bb & m0) << 1u;
        bb = swap_bits<uint64_t, m1, 4u>(bb);
        bb = swap_bits<uint64_t, m2, 8u>(bb);
        bb = swap_bits<uint64_t, m3, 20u>(bb);
        bb = (bb >> 34u) | (bb << 30u);
        return bb;
    }

    inline Bitboard mirror_files(Bitboard bb) {
        assert(BOARD_IS_8x8);
        static constexpr Bitboard k1 = 0x5555555555555555ULL;
        static constexpr Bitboard k2 = 0x3333333333333333ULL;
        static constexpr Bitboard k4 = 0x0f0f0f0f0f0f0f0fULL;
        bb = ((bb >> 1u) & k1) | ((bb & k1) << 1u);
        bb = ((bb >> 2u) & k2) | ((bb & k2) << 2u);
        bb = ((bb >> 4u) & k4) | ((bb & k4) << 4u);
        return bb;
    }
#endif //USE_BUILTIN_MACROS

}

#endif //ABG8X8_BITBOARD_H
