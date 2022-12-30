//
// Created by Robin on 19.01.2018.
//

#include "MovePicker.h"
#include <assert.h>
namespace Statistics {

MovePicker mPicker;



void MovePicker::init() {

//	policy.addLayer(Layer{120, 1024});
//    policy.addLayer(Layer{1024, 8});
//    policy.addLayer(Layer{8, 32});
//    policy.addLayer(Layer{32, 128});
//    policy.load("policy.quant");
//    policy.init();

}


int MovePicker::get_move_encoding(Move move) {
//    if (color == BLACK) {
//        move.from = getMirrored(move.from);
//        move.to = getMirrored(move.to);
//    }
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
    return 4*move.get_from_index()+dir;
}

int MovePicker::get_history_index(Position pos, Move move) {
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



void MovePicker::clear_scores() {
    std::fill(history.begin(), history.end(), 0);
    for(auto i=0; i<killer_moves.size(); ++i) {
        for(auto k=0; k<MAX_KILLERS; ++k) {
            killer_moves[i][k]=Move{};
        }
    }
    for(auto i=0; i<counter_history.size(); ++i) {
        for(auto k=0; k<counter_history[0].size(); ++k) {
            counter_history[i][k]=0;
        }
    }
}
int MovePicker::get_move_score(Position pos, Move move, Move previous, Depth depth)
{
    // const int index = get_move_encoding(pos.get_color(),move);
    const int index = get_history_index(pos,move);
    int score = history[index];
    if(!previous.is_capture() && !move.is_capture()) {
        auto counter=counter_history[get_history_index(pos,previous)][get_move_encoding(move)];
       score+=counter;
    }
//	auto pol = policy[index];
//	return pol;
    return score;

}

int MovePicker::get_move_score(Position current, Depth depth, int ply, Move move,Move previous, Move ttMove) {
    if (move == ttMove) {
        return std::numeric_limits<int32_t>::max();
    }

    if(move == killer_moves[ply][1] ||move == killer_moves[ply][0]) {
        return std::numeric_limits<int32_t>::max()-1000;
    }

    if (move.is_capture()) {
        return (int) Bits::pop_count(move.captures);
    }

    return get_move_score(current, move,previous, depth);

}

void update_history_score(int& score, int delta) {
    score+=delta;
}


void MovePicker::update_scores(Position pos, Move *liste, Move move,Move previous, int depth) {
    const int index = get_history_index(pos, move);
    const int delta = std::min(depth*depth,14*14);;
    update_history_score(history[index],delta);
    Move top = liste[0];

	
        if(!previous.is_capture() && !move.is_capture()) {
            counter_history[get_history_index(pos,previous)][get_move_encoding(move)]+=delta;
        }

    while (top != move) {
        if (top == move)
            break;
        top = *liste;
        int& score = history[get_history_index(pos, top)];
        update_history_score(score,-delta);


        if(!previous.is_capture() && !top.is_capture()) {
            counter_history[get_history_index(pos,previous)][get_move_encoding(top)]-=delta;
        }

        liste++;
    }
}
}
