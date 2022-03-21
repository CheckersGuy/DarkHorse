//
// Created by robin on 21.03.22.
//

#ifndef READING_COMPRESS_H
#define READING_COMPRESS_H

#include <fstream>
#include <Position.h>
#include <MoveListe.h>
#include <MGenerator.h>

template<typename Iter>
void compress_game(Iter begin, Iter end, std::ofstream &stream) {
    //storing the starting position and then storing the index of the moves made
    const Position start_pos = *begin;
    stream << start_pos;
    begin++;
    Position previous = start_pos;
    std::vector<uint8_t> indices;
    std::for_each(begin, end, [&](Position next) {
        MoveListe liste;
        get_moves(previous, liste);
        uint8_t move_index;
        for (int i = 0; i < liste.length(); ++i) {
            Position copy = previous;
            copy.make_move(liste[i]);
            if (copy == next) {
                move_index = i;
                break;
            }
        }
        indices.emplace_back(move_index);
        previous = next;
    });
    //storing the number of indices
    const uint8_t num_indices = indices.size();
    stream.write((char *) &num_indices, sizeof(uint8_t) * num_indices);
    //now we can store the indices
    stream.write((char *) &indices[0], sizeof(uint8_t) * num_indices);


}

template<typename OutIter>
void read_compressed_game(OutIter iter, std::ifstream &stream) {
    Position start_pos;
    stream >> start_pos;
    uint8_t num_indices;
    stream.read((char *) &num_indices, sizeof(uint8_t));
}


#endif //READING_COMPRESS_H
