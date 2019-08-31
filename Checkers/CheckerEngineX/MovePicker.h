//
// Created by Robin on 19.01.2018.
//

#ifndef CHECKERENGINEX_MOVEPICKER_H
#define CHECKERENGINEX_MOVEPICKER_H

#include "types.h"
#include "Move.h"


namespace Statistics {
    class MovePicker {
    private:
        int history[32 * 32 * 2] = {0};
        int bfScore[32 * 32 * 2] = {0};
    public:
        int getMoveScore(Move move, Color color);

        int getMoveScore(Move move, Color color, Move ttMove);

        void clearScores();

        void updateHHScore(Move move, Color color, int depth);

        void updateBFScore(Move *liste, int moveIndex, Color color, int depth);
    };


    extern MovePicker mPicker;


}
#endif //CHECKERENGINEX_MOVEPICKER_H
