//
// Created by robin on 21.03.22.
//

#ifndef READING_COMPRESS_H
#define READING_COMPRESS_H

#include <fstream>
#include <Position.h>
#include <MoveListe.h>
#include <MGenerator.h>

struct Game {
    Position start_position;
    Position previous;
    std::array<uint8_t, 600> indices;
    uint16_t num_indices{0};

    Game(const Position start_position) : start_position(start_position) {
        previous = start_position;
    }

    void add_index(uint8_t index) {
        indices[num_indices++] = index;
        MoveListe liste;
        get_moves(previous, liste);
        previous.make_move(liste[index]);
    }

    void add_position(Position pos) {
        MoveListe liste;
        get_moves(previous, liste);
        for (auto i = 0; i < liste.length(); ++i) {
            Position t = previous;
            t.make_move(liste[i]);
            if (t == pos) {
                indices[num_indices++] = i;
                break;
            }
        }
    }

    //overloading read write operators

    friend std::ifstream operator>>(std::ifstream &stream, Game &game) {
        Position start_pos;
        stream >> start_pos;
        start_pos.print_position();
        stream.read((char *) &game.num_indices, sizeof(uint16_t));
        Position current = start_pos;
        std::cout << (int) game.num_indices << std::endl;
        stream.read((char *) &game.indices[0], sizeof(uint8_t) * game.num_indices);
        for (auto i = 0; i < game.num_indices; ++i) {
            MoveListe liste;
            get_moves(start_pos, liste);
            start_pos.make_move(liste[game.indices[i]]);
        }
    }

    friend std::ofstream operator<<(std::ofstream &stream, const Game &game) {
        Position start_pos = game.start_position;
        stream << start_pos;
        const uint16_t num_indices = game.indices.size();
        stream.write((char *) &num_indices, sizeof(uint16_t));
        stream.write((char *) &game.indices[0], sizeof(uint8_t) * num_indices);
    }
};

template<typename Iter>
void compress_game(Iter begin, Iter end, std::ofstream &stream) {
    //storing the starting position and then storing the index of the moves made
    Position start_pos = Position::get_start_position();
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
    const uint16_t num_indices = indices.size();

    stream.write((char *) &num_indices, sizeof(uint16_t));
    std::cout << "Test: " << num_indices << std::endl;
    //now we can store the indices
    stream.write((char *) &indices[0], sizeof(uint8_t) * num_indices);
}

template<typename OutIter>
void read_compressed_game(OutIter iter, std::ifstream &stream) {
    Position start_pos;

    stream >> start_pos;
    start_pos.print_position();

    uint16_t num_indices;
    stream.read((char *) &num_indices, sizeof(uint16_t));
    std::cout << "NumIndices: " << (int) num_indices << std::endl;
    Position current = start_pos;
    std::cout << (int) num_indices << std::endl;
    std::unique_ptr<uint8_t[]> indices = std::make_unique<uint8_t[]>(num_indices);
    stream.read((char *) indices.get(), sizeof(uint8_t) * num_indices);
    for (auto i = 0; i < num_indices; ++i) {
        MoveListe liste;
        get_moves(start_pos, liste);
        start_pos.make_move(liste[indices[i]]);
    }

}


#endif //READING_COMPRESS_H
