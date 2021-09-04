//
// Created by robin on 01.09.21.
//

#ifndef READING_SAMPLE_H
#define READING_SAMPLE_H

#include "Position.h"
#include <random>

struct Sample {
    Position position;
    int result{1000};

    friend std::ostream &operator<<(std::ostream &stream, const Sample s);

    friend std::istream &operator>>(std::istream &stream, Sample &s);

    bool operator==(const Sample &other) const;

    bool operator!=(const Sample &other) const;
};

namespace std {

    template<>
    struct hash<Sample> {
        std::hash<int> hasher;
        std::array<std::array<uint64_t, 4>, 32> keys;
        uint64_t color_black;
        //should not use the existing zobrist keys

        hash() {
            std::mt19937 generator(23123123ull);
            std::uniform_int_distribution<uint64_t> distrib;
            for (int i = 0; i < 32; ++i) {
                for (int j = 0; j < 4; ++j) {
                    const auto value = distrib(generator);
                    keys[i][j] = value;
                }
            }
            color_black = distrib(generator);
        }

        uint64_t operator()(const Sample &s) const {
            const uint32_t BK = s.position.K & s.position.BP;
            const uint32_t WK = s.position.K & s.position.WP;
            uint64_t nextKey = 0u;
            uint32_t allPieces = s.position.BP | s.position.WP;
            while (allPieces) {
                uint32_t index = Bits::bitscan_foward(allPieces);
                uint32_t maske = 1u << index;
                if ((maske & BK)) {
                    nextKey ^= keys[index][BKING];
                } else if ((maske & s.position.BP)) {
                    nextKey ^= keys[index][BPAWN];
                }
                if ((maske & WK)) {
                    nextKey ^= keys[index][WKING];
                } else if ((maske & s.position.WP)) {
                    nextKey ^= keys[index][WPAWN];
                }
                allPieces &= allPieces - 1u;
            }
            if (s.position.color == BLACK) {
                nextKey ^= color_black;
            }
            return nextKey ^ hasher(s.result);
        }
    };

}


#endif //READING_SAMPLE_H
