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
    Result result{Result::UNKNOWN};
    Result tb_result{Result::UNKNOWN};

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

    void set_result(Result res) {
        result = res;
    }

    void add_position(Position pos) {
        //need to be consecutive positions

        if (num_indices == 0) {
            start_position = pos;
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
        stream.read((char *) &game.result, sizeof(Result));
        stream.read((char *) &game.tb_result, sizeof(Result));
        stream.read((char *) &game.indices[0], sizeof(uint8_t) * game.num_indices);
        return stream;
    }

    friend std::ofstream &operator<<(std::ofstream &stream, const Game &game) {
        Position start_pos = game.start_position;
        stream << start_pos;
        stream.write((char *) &game.num_indices, sizeof(uint16_t));
        stream.write((char *) &game.result, sizeof(Result));
        stream.write((char *) &game.tb_result, sizeof(Result));
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
        beg.index = num_indices + 1;
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

//converting old to new training format

void convert_to_new(std::string input, std::string output) {
    std::ofstream out_stream(output);
    std::ifstream in_stream(input);
    std::istream_iterator<Sample> begin(in_stream);
    std::istream_iterator<Sample> end{};
    //to be continued

    size_t piece_count = 32u;
    Game game;
    std::for_each(begin, end, [&](Sample s) {
        if (Bits::pop_count(s.position.BP | s.position.WP) <= piece_count) {
            game.add_position(s.position);
            piece_count = Bits::pop_count(s.position.WP | s.position.BP);
        } else {
            piece_count = 32;
            out_stream << game;
        }


    });


}


#endif //READING_COMPRESS_H
