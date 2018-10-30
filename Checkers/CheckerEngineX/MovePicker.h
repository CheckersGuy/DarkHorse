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
        int history[32 * 32 * 2] = {-10};
        int bfScore[32 * 32 * 2] = {-10};
    public:
        int getMoveScore(Move move, Color color);

        int getMoveScore(Move move, Color color, Move ttMove);

        int getHHScore(Move move, Color color);

        int getBFScore(Move move, Color color);

        void clearScores();

        void updateHHScore(Move move, Color color, int depth);

        void updateBFScore(Move *liste, int moveIndex, Color color, int depth);
    };

#ifdef TRAIN
    extern thread_local MovePicker mPicker;
#else
    extern MovePicker mPicker;
#endif



}
#endif //CHECKERENGINEX_MOVEPICKER_H
