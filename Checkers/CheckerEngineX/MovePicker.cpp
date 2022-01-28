//
// Created by Robin on 19.01.2018.
//

#include "MovePicker.h"

namespace Statistics {

    MovePicker mPicker;

    void MovePicker::init() {
    /*    policy.load("policy.weights");
        policy.addLayer(Layer{120, 256});
        policy.addLayer(Layer{256, 32});
        policy.addLayer(Layer{32, 32});
        policy.addLayer(Layer{32, 100});

        policy.init();*/

    }

    int MovePicker::get_move_encoding(Color color, Move move) {

        if (color == BLACK) {
            move.from = getMirrored(move.from);
            move.to = getMirrored(move.to);
        }

        const uint32_t maske = OUTER_SQUARES;


        if ((move.from & maske) != 0) {
            uint32_t index = Bits::pext(move.from, maske);
            index = _tzcnt_u32(index);
            int dir;
            if ((((move.to & MASK_L3) << 3) == move.from) || (((move.to) << 4) == move.from) ||
                (((move.to & MASK_L5) << 5) == move.from)) {
                dir = 0;
            } else {
                dir = 1;
            }
            return 2 * index + dir;
        }

        const uint32_t maske2 = PROMO_SQUARES_BLACK | PROMO_SQUARES_WHITE;

        if ((move.from & maske2) != 0) {
            uint32_t index = Bits::pext(move.from, maske2);
            index = _tzcnt_u32(index);
            int dir;
            if ((((move.to & MASK_L3) << 3) == move.from) || (((move.to & MASK_L5) << 5) == move.from)) {
                dir = 0;
            } else if (((move.to) << 4) == move.from) {
                dir = 1;
            } else if (((move.to) >> 4) == move.from) {
                dir = 0;
            } else if ((((move.to & MASK_R3) >> 3) == move.from) || (((move.to & MASK_R5) >> 5) == move.from)) {
                dir = 1;
            }
            return 12 + 2 * index + dir;
        }
        const uint32_t maske3 = INNER_SQUARES;

        if ((move.from & maske3) != 0) {
            uint32_t index = Bits::pext(move.from, maske3);
            index = _tzcnt_u32(index);
            int dir;
            if ((((move.to & MASK_L3) << 3) == move.from) || (((move.to & MASK_L5) << 5) == move.from)) {
                dir = 0;
            } else if (((move.to) << 4) == move.from) {
                dir = 1;
            } else if (((move.to) >> 4) == move.from) {
                dir = 2;
            } else if ((((move.to & MASK_R3) >> 3) == move.from) || (((move.to & MASK_R5) >> 5) == move.from)) {
                dir = 3;
            }

            return 12 + 16 + 4 * index + dir;

        }


        return -1;
    }

    int MovePicker::getHistoryIndex(Position pos, Move move) {
        //32 source squares
        // 4 piece types
        // 4 directions
        int t;

        if ((move.from & (pos.BP & pos.K)) != 0) {
            t = 0;
        } else if ((move.from & (pos.WP & pos.K)) != 0) {
            t = 1;
        } else if ((move.from & pos.BP) != 0) {
            t = 2;
        } else if ((move.from & pos.WP) != 0) {
            t = 3;
        } else {
            t = 0;
        }

        int orig_sq = move.get_from_index();
        //direction of the piece
        int dir = 0;

        if ((((MASK_R3 & move.to) >> 3) == move.from) || (((MASK_R5 & move.to) >> 5) == move.from)) {
            dir = 1;
        } else if ((((MASK_L3 & move.to) << 3) == move.from) || (((MASK_L5 & move.to) << 5) == move.from)) {
            dir = 2;
        } else if ((move.to << 4) == move.from) {
            dir = 3;
        }
        const int index = 16 * orig_sq + 4 * dir + t;
        return index;
    }

    void MovePicker::clearScores() {
        std::fill(history.begin(), history.end(), 0);
    }

    int MovePicker::getMoveScore(Position pos, Move move, Depth depth) {
            const int index = getHistoryIndex(pos, move);
            const int score = history[index];
            const int bf_score = bfScore[index] + 1;
            return score;


  /*      auto score = policy.get_output()[get_move_encoding(pos.get_color(), move)] * 1000;
        return score;*/
    }

    int MovePicker::getMoveScore(Position current, Depth depth, int ply, Move move, Move ttMove) {
        if (move == ttMove) {
            return std::numeric_limits<int16_t>::max();
        }
        if (move.is_capture()) {
            return (int) Bits::pop_count(move.captures);
        }

        return getMoveScore(current, move,depth);

    }


    void MovePicker::update_scores(Position pos, Move *liste, Move move, int depth) {
        if (depth <= 2)
            return;
        const int index = getHistoryIndex(pos, move);
        history[index] += depth * depth;
        Move top = liste[0];
        while (top != move) {
            if (top == move)
                break;
            top = *liste;
            history[getHistoryIndex(pos, top)] -= depth * depth;
            liste++;
        }
    }
}