//
// Created by robin on 5/21/18.
//

#include "Utilities.h"

namespace Utilities {

    std::unordered_set<uint64_t> hashes;

    void prepare_training_set(Training::TrainData &data) {
        auto invert = [](Training::Position &pos) {
            //inverts all positions where it is black to move
            //so in every position it's white to move
            //that makes training using tensorflow a little easier
            auto orig_color = pos.mover();
            Position temp;
            temp.BP = pos.bp();
            temp.WP = pos.wp();
            temp.K = pos.k();
            temp.color = (pos.mover()==Training::BLACK) ? BLACK : WHITE;
            //flipping the board


            if (temp.color == BLACK) {
                temp = temp.getColorFlip();
            }
            pos.set_bp(temp.BP);
            pos.set_wp(temp.WP);
            pos.set_k(temp.K);
            pos.set_mover(orig_color);
            //flips only the position but not the color
            //that way I can compute the weights w.r.t the player to move
        };
        std::for_each(data.mutable_positions()->begin(), data.mutable_positions()->end(), invert);

    }

}