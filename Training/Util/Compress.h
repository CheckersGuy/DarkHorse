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
    std::array<uint8_t, 600> indices;
    uint16_t num_indices{0};
    Result result{Result::UNKNOWN};

    Game(const Position start_position) : start_position(start_position) {
    }

    Game() = default;

    void add_move(uint8_t move_index) {
        Game::encode_move(indices[num_indices++], move_index);
    }


    void set_result(Result res) {
        result = res;
    }

    void set_result(Result res, int n) {
        //sets the result of the nth position
        Game::encode_result(indices[n], res);
    }

    void add_position(Position pos) {
        //need to be consecutive positions

        if (start_position.is_empty()) {
            start_position = pos;
            return;
        }

        MoveListe liste;
        Position x = get_last_position();
        get_moves(x, liste);
        for (auto i = 0; i < liste.length(); ++i) {
            Position t = x;
            t.make_move(liste[i]);
            if (t == pos) {
                Game::encode_move(indices[num_indices++], i);
                break;
            }
        }
    }

    //overloading read write operators

    friend std::ifstream &operator>>(std::ifstream &stream, Game &game) {
        stream >> game.start_position;
        stream.read((char *) &game.num_indices, sizeof(uint16_t));
        stream.read((char *) &game.result, sizeof(Result));
        stream.read((char *) &game.indices[0], sizeof(uint8_t) * game.num_indices);
        return stream;
    }

    friend std::ofstream &operator<<(std::ofstream &stream, const Game &game) {
        Position start_pos = game.start_position;
        stream << start_pos;
        stream.write((char *) &game.num_indices, sizeof(uint16_t));
        stream.write((char *) &game.result, sizeof(Result));
        stream.write((char *) &game.indices[0], sizeof(uint8_t) * game.num_indices);
        return stream;
    }

    Position get_position(int n) {
        //returns the position after the nth move
        Position current = start_position;
        for (auto i = 0; i < n; ++i) {
            MoveListe liste;
            get_moves(current, liste);
            current.make_move(liste[Game::get_move_index(indices[i])]);
        }
        return current;
    }

    Position get_last_position() {
        return get_position(num_indices);
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
        beg.index = num_indices + 1;
        return beg;
    }


    static void encode_move(uint8_t &bit_field, uint8_t move_index) {
        //clearing the move field
        bit_field &= ~(63);
        bit_field |= move_index;
    }

    static void encode_result(uint8_t &bit_field, Result result) {
        uint8_t temp = static_cast<uint8_t>(result);
        const uint8_t clear_bits = 3ull << 6;
        bit_field &= ~clear_bits;
        bit_field |= temp << 6;
    }

    static uint8_t get_move_index(uint8_t &bit_field) {
        const uint8_t clear_bits = 3ull << 6;
        uint8_t copy = bit_field & (~clear_bits);
        return copy;
    }

    static Result get_result(uint8_t &bit_field) {
        uint8_t copy = (bit_field >> 6) & 3ull;
        return static_cast<Result>(copy);
    }


    template<typename OutIter>
    void extract_samples(OutIter iterator) {
        Sample sample;
        sample.position = start_position;
        sample.result = Game::get_result(indices[0]);
        sample.move = -1;
        *iterator = sample;
        iterator++;
        for (auto i = 1; i < num_indices; ++i) {
            Position current = get_position(i);
            sample.position = current;
            sample.result = Game::get_result(indices[i]);
            sample.move = -1;
            *iterator = sample;
            iterator++;
        }
        //last position as well
        Position current = get_position(num_indices);
        sample.position = current;
        sample.result = result;
        sample.move = -1;
        *iterator = sample;
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

//not trying to convert to the new format
//will change game generation and generate new data
//requires changing rescoring too though




#endif //READING_COMPRESS_H
