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


int MovePicker::get_move_encoding(Color color, Move move) {
    if (color == BLACK) {
        move.from = getMirrored(move.from);
        move.to = getMirrored(move.to);
    }
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
}

int MovePicker::get_move_score(Position pos, Move move, Depth depth)
{
   // const int index = get_move_encoding(pos.get_color(),move);
    const int index = get_history_index(pos,move);
	int score = history[index];
//	auto pol = policy[index];
//	return pol;
    return score;

}

int MovePicker::get_move_score(Position current, Depth depth, int ply, Move move, Move ttMove) {
    if (move == ttMove) {
        return std::numeric_limits<int32_t>::max();
    }

    if (move.is_capture()) {
        return (int) Bits::pop_count(move.captures);
    }

    return get_move_score(current, move, depth);

}

void update_history_score(int& score, int delta){
		score+=delta;
}


void MovePicker::update_scores(Position pos, Move *liste, Move move, int depth) {
    const int index = get_history_index(pos, move);
	const int delta = depth;
	update_history_score(history[index],delta);
    Move top = liste[0];
    while (top != move) {
        if (top == move)
            break;
        top = *liste;
        int& score = history[get_history_index(pos, top)];
		update_history_score(score,-delta);
        liste++;
    }
}
}
