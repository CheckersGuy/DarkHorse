//
// Created by Robin on 19.01.2018.
//

#include "MovePicker.h"

namespace Statistics {

    MovePicker mPicker;

    int MovePicker::get_move_encoding(Color color, Move move) {
        //we differentiate between edge moves and
        //always from whites perspective
        return 0;
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

        int orig_sq = move.getFromIndex();
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

    int MovePicker::getMoveScore(Position pos, Move move) {
        const int index = getHistoryIndex(pos, move);
        const int score = history[index];
        const int bf_score = bfScore[index] + 1;
        return score;
    }

    int MovePicker::getMoveScore(Position current, int ply, Move move, Move ttMove) {
        if (move == ttMove) {
            return std::numeric_limits<int16_t>::max();
        }
        if (move.isCapture()) {
            return (int) Bits::pop_count(move.captures);
        }


        /*      if (move == killer_moves[ply]) {
                  return killer_score;
              }
      */

        return getMoveScore(current, move);

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