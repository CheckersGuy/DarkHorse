//
// Created by robin on 21.03.22.
//

#ifndef READING_COMPRESS_H
#define READING_COMPRESS_H

#include <fstream>
#include <Position.h>
#include <MoveListe.h>
#include <MGenerator.h>

struct Game;

struct GameIterator {
    Game &game;
    int index{0};

    GameIterator(const GameIterator &other) : game(other.game) {
        index = other.index;
    }

    GameIterator(Game &game) : game(game) {

    }

    bool operator==(GameIterator &other);

    bool operator!=(GameIterator &other);

    GameIterator &operator++() {
        index++;
        return *this;
    }

    GameIterator &operator--() {
        index--;
        return *this;
    }

    GameIterator operator++(int) {
        GameIterator copy(*this);
        index++;
        return copy;
    }

    GameIterator operator--(int) {
        GameIterator copy(*this);
        index--;
        return copy;
    }

    Position operator*();

};

struct Game {
    Position start_position;
    Position previous;
    std::array<uint8_t, 600> indices;
    uint16_t num_indices{0};

    Game(const Position start_position) : start_position(start_position) {
        previous = start_position;
    }

    Game() = default;

    void add_index(uint8_t index) {
        indices[num_indices++] = index;
        MoveListe liste;
        get_moves(previous, liste);
        previous.make_move(liste[index]);
    }

    void add_position(Position pos) {
        //need to be consecutive positions

        if(num_indices ==0){
            start_position =pos;
            return;
        }

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

    friend std::ifstream &operator>>(std::ifstream &stream, Game &game) {
        stream >> game.start_position;
        stream.read((char *) &game.num_indices, sizeof(uint16_t));
        std::cout << (int) game.num_indices << std::endl;
        stream.read((char *) &game.indices[0], sizeof(uint8_t) * game.num_indices);
        return stream;
    }

    friend std::ofstream &operator<<(std::ofstream &stream, const Game &game) {
        Position start_pos = game.start_position;
        stream << start_pos;
        stream.write((char *) &game.num_indices, sizeof(uint16_t));
        stream.write((char *) &game.indices[0], sizeof(uint8_t) * game.num_indices);
        return stream;
    }

    Position get_position(int n) {
        //returns the position after the nth move
        Position current = start_position;
        for (auto i = 0; i < n; ++i) {
            MoveListe liste;
            get_moves(current, liste);
            current.make_move(liste[indices[i]]);
        }
        return current;
    }

    bool operator==(Game &other) {
        return (other.start_position == start_position &&
                std::equal(indices.begin(), indices.end(), other.indices.begin()));
    }

    bool operator!=(Game &other) {
        return !((*this) == other);
    }

    GameIterator begin() {
        GameIterator beg(*this);
        return beg;
    }

    GameIterator end() {
        GameIterator beg(*this);
        beg.index = num_indices;
        return beg;
    }
};

Position GameIterator::operator*() {
    Position current = game.get_position(index);
    return current;
}

bool GameIterator::operator==(GameIterator &other) {
    return (other.game == game && other.index == index);
}

bool GameIterator::operator!=(GameIterator &other) {
    return (other.game != game || other.index != index);
}


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


#endif //READING_COMPRESS_H
