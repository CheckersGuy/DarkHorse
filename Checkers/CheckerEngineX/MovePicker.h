//
// Created by Robin on 19.01.2018.
//

#ifndef CHECKERENGINEX_MOVEPICKER_H
#define CHECKERENGINEX_MOVEPICKER_H

#include "types.h"
#include "Move.h"
#include "Position.h"
#include "Network.h"
#include "MoveListe.h"
#include "Move.h"
namespace Statistics {
    class MovePicker {
    private:
      
        std::array<int, 32 * 16> history{0};
        std::array<int, 32 * 16> bfScore{0};
      
    public:
        std::array<std::array<Move,2>,MAX_PLY> killer_moves;
        int get_move_score(Position pos, Move move, Depth depth);

        int get_history_index(Position pos, Move move);

        int get_move_score(Position current, Depth depth, int ply, Move move, Move ttMove);

        void clear_scores();

        void update_scores(Position pos, Move *list, Move move, int depth);
        
        static int get_move_encoding(Color color, Move move);

        void init();

    };

    extern MovePicker mPicker;


}
#endif //CHECKERENGINEX_MOVEPICKER_H
