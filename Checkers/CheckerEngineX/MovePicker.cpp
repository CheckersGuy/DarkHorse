//
// Created by Robin on 19.01.2018.
//

#include "MovePicker.h"

namespace Statistics {


#ifndef TRAIN
    MovePicker mPicker;
#else
    thread_local MovePicker mPicker;
#endif

    void MovePicker::init() {
    /*    policy_net.load("policytestreinf");
        policy_net.addLayer(Layer{120, 1024});
        policy_net.addLayer(Layer{1024, 64});
        policy_net.addLayer(Layer{64, 64});
        policy_net.addLayer(Layer{64, 32});
        policy_net.init();*/
    }

    int MovePicker::get_move_encoding(Color color, Move move) {
        //we differentiate between edge moves and
        //always from whites perspective

        auto cmp_class = [](uint32_t from, uint32_t to) {
            //outputs a value between 0 and 4 depending on
            //how we can reach the destination square
            if (to == from << 3 || to == from << 5) {
                return 0;
            }
            if (to == from >> 3 || to == from >> 5) {
                return 1;
            }
            if (to == from << 4)
                return 2;
            if (to == from >> 4)
                return 3;

        };

        const uint32_t edge = 0u;

        uint32_t from = (color == BLACK) ? getMirrored(move.from) : move.from;
        uint32_t to = (color == BLACK) ? getMirrored(move.to) : move.to;
        //we classify moves based on that
        //EDGE moves first

        //getting the index on one of the edges
        uint32_t current = edge & from;
        current = _pext_u32(current, edge);
        uint32_t move_index = Bits::bitscan_foward(current);

        //to be continued


        return 0;
    }

    void MovePicker::clearScores() {
        std::fill(bfScore.begin(), bfScore.end(), 0);
        std::fill(history.begin(), history.end(), 0);
    }

    int MovePicker::getMoveScore(Move move, Color color) {
        const int colorIndex = (color + 1) / 2;
        const int bScore = bfScore[32 * 32 * colorIndex + 32 * move.getToIndex() + move.getFromIndex()] + 1;
        const int hhScore = history[32 * 32 * colorIndex + 32 * move.getToIndex() + move.getFromIndex()];
        const int score = hhScore / bScore;

        return score;
    }

    int MovePicker::getMoveScore(Position current, Depth depth, Move move, Color color, Move ttMove) {

        if (move == ttMove) {
            return std::numeric_limits<int16_t>::max();
        }

        int score = getMoveScore(move, color);
   /*    if (depth >= 10) {
            policy_net.compute_incre_forward_pass(current);
            uint32_t temp = (current.color == BLACK) ? getMirrored(move.from) : move.from;
            uint32_t idx = Bits::bitscan_foward(temp);
            score = (int) (policy_net.input[idx])+ getMoveScore(move, color);;
            return score;
        }*/
        return score;
        /*  if (move.isCapture()) {
              uint32_t king_cap, pawn_cap;
              king_cap = Bits::pop_count(current.K & move.captures);
              pawn_cap = Bits::pop_count((current.WP | current.BP) & move.captures & (~current.K));

              return 10 * (int) pawn_cap + 12 * (int) king_cap;
          }
*/
        return getMoveScore(move, color);

    }


    void MovePicker::update_scores(Move *liste, Move move, Color color, int depth) {
        const int colorIndex = (color + 1) / 2;
        history[32 * 32 * colorIndex + 32 * move.getToIndex() + move.getFromIndex()] += depth * depth;
        while (move != (*liste)) {
            Move top = *liste;
            bfScore[32 * 32 * colorIndex + 32 * top.getToIndex() + top.getFromIndex()] += depth;
            liste++;
        }

    }
}