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
        std::array<int, 32 * 32 * 2> history{0};
        std::array<int, 32 * 32 * 2> bfScore{0};
    public:

        int getMoveScore(Move move, Color color);

        int getMoveScore(Move move, Color color, Move ttMove);

        void clearScores();

        void update_scores(Move* list, Move move,Color color, int depth);

    };


    extern MovePicker mPicker;


}
#endif //CHECKERENGINEX_MOVEPICKER_H
