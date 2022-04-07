//
// Created by robin on 21.03.22.
//

#ifndef READING_COMPRESS_H
#define READING_COMPRESS_H

#include <fstream>
#include <Position.h>
#include <MoveListe.h>
#include <MGenerator.h>
#include <regex>
#include <filesystem>
#include  "../BloomFilter.h"
#include <map>

struct Game;

struct GameIterator {
    Game &game;
    int index{0};
    using iterator_category = std::forward_iterator_tag;
    using difference_type = size_t;
    using value_type = Position;
    using pointer = Position *;  // or also value_type*
    using reference = Position &;  // or also value_type&

    GameIterator(const GameIterator &other) : game(other.game) {
        index = other.index;
    }

    GameIterator(Game &game) : game(game) {

    }

    bool operator==(GameIterator &other) const;

    bool operator!=(GameIterator &other) const;

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

    friend std::istream &operator>>(std::istream &stream, Game &game) {
        stream >> game.start_position;
        stream.read((char *) &game.num_indices, sizeof(uint16_t));
        stream.read((char *) &game.result, sizeof(Result));
        stream.read((char *) &game.indices[0], sizeof(uint8_t) * game.num_indices);
        return stream;
    }

    friend std::ostream &operator<<(std::ostream &stream, const Game &game) {
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
        MoveListe liste;
        Sample sample;
        sample.position = start_position;

        get_moves(sample.position,liste);

        sample.result = Game::get_result(indices[0]);
        sample.move = Statistics::mPicker.get_move_encoding(sample.position.get_color(),liste[Game::get_move_index(indices[0])]);
        *iterator = sample;
        iterator++;
        for (auto i = 1; i < num_indices; ++i) {
            liste = MoveListe{};
            Position current = get_position(i);
            sample.position = current;
            get_moves(sample.position,liste);
            sample.result = Game::get_result(indices[i]);
            sample.move = Statistics::mPicker.get_move_encoding(sample.position.get_color(),liste[Game::get_move_index(indices[i])]);
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

inline Position GameIterator::operator*() {
    Position current = game.get_position(index);
    return current;
}

inline bool GameIterator::operator==(GameIterator &other) const {
    return (other.game == game && other.index == index);
}

inline bool GameIterator::operator!=(GameIterator &other) const {
    return (other.game != game || other.index != index);
}

//not trying to convert to the new format
//will change game generation and generate new data
//requires changing rescoring too though

template<typename Iterator>
inline void merge_training_data(Iterator begin, Iterator end, std::string output) {
    std::ofstream stream_out(output, std::ios::app | std::ios::binary);
    for (auto it = begin; it != end; ++it) {
        auto file_input = *it;
        std::ifstream stream(file_input.c_str());
        Game game;
        while (stream >> game) {
            stream_out << game;
        }
    }
    //this should merge alle the files into one
}


inline std::optional<std::string> is_temporary_train_file(std::string name) {
    std::regex reg("[a-z0-9\\-\\_]+[.]train[.]temp[0-9]+");
    if (std::regex_match(name, reg)) {
        auto f = name.find('.');
        return std::make_optional(name.substr(0, f));
    }
    return std::nullopt;
}

template<typename Iterator>
inline std::pair<size_t, size_t> count_unique_positions(Iterator begin, Iterator end) {
    BloomFilter<Position> filter(9585058378, 3);
    size_t unique_count = 0;
    size_t total_positions = 0;
    for (auto it = begin; it != end; ++it) {
        Game game = (*it);
        for (auto pos: game) {
            if (!filter.has(pos)) {
                unique_count++;
                filter.insert(pos);
            }
            total_positions++;
        }

    }
    return std::make_pair(unique_count, total_positions);
}

inline std::pair<size_t, size_t> count_unique_positions(std::string game_file) {
    std::ifstream stream(game_file, std::ios::binary);
    std::istream_iterator<Game> begin(stream);
    std::istream_iterator<Game> end;
    return count_unique_positions(begin, end);
}

inline size_t count_trainable_positions(std::string game_file) {
    std::ifstream stream(game_file, std::ios::binary);
    std::istream_iterator<Game> begin(stream);
    std::istream_iterator<Game> end;
    size_t counter{0};
    //std::vector<Sample> samples;
    std::for_each(begin, end, [&](Game g) {
     /*   samples.clear();
        g.extract_samples(std::back_inserter(samples));
        counter += std::count_if(samples.begin(), samples.end(), [](Sample s) {
            return s.result != UNKNOWN && !s.position.has_jumps();
        });*/
     counter+=g.num_indices+1;

    });
    return counter;
}

inline void merge_temporary_files(std::string directory, std::string out_directory) {
    std::filesystem::path in_path(directory);
    std::filesystem::path out_path(out_directory);
    if (!std::filesystem::is_directory(in_path)) {
        throw std::string{"Not a directory as in_path"};
    }
    if (!std::filesystem::is_directory(out_path)) {
        throw std::string{"Not a directory as out_path"};
    }
    //finding all relevant files
    std::map<std::string, std::vector<std::filesystem::path>> my_map;
    for (auto &ent: std::filesystem::directory_iterator(in_path)) {
        auto path = ent.path();
        auto t = is_temporary_train_file(path.filename().string());
        if (t.has_value()) {
            my_map[t.value()].emplace_back(path);
        }
    }
    //listing all the temporary files found
    for (auto &val: my_map) {
        std::cout << val.first << std::endl;
        std::vector<std::filesystem::path> local_paths;

        for (auto &path_file: val.second) {
            local_paths.emplace_back(path_file.c_str());
        }
        merge_training_data(local_paths.begin(), local_paths.end(), out_directory + val.first + ".train");

        for (auto &path_file: val.second) {
            std::filesystem::remove_all(path_file);
        }
    }

}

#endif //READING_COMPRESS_H
