//
// Created by Robin on 19.01.2018.
//

#ifndef CHECKERENGINEX_MOVEPICKER_H
#define CHECKERENGINEX_MOVEPICKER_H

#include "types.h"
#include "Move.h"
#include "Position.h"
#include "Network.h"
namespace Statistics {
    class MovePicker {
    private:
        Network policy_net;
        std::array<int, 32 * 32 * 2> history{0};
        std::array<int, 32 * 32 * 2> bfScore{0};
    public:

        int getMoveScore(Move move, Color color);

        int getMoveScore(Position current,Depth depth,Move move, Color color, Move ttMove);

        void clearScores();

        void init();

        void update_scores(Move* list, Move move,Color color, int depth);

        static int get_move_encoding(Color color,Move move);

    };

#ifndef TRAIN
    extern MovePicker mPicker;
#else
    extern thread_local MovePicker mPicker;
#endif



}
#endif //CHECKERENGINEX_MOVEPICKER_H
