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
#include "../BloomFilter.h"
#include <map>
#include <vector>

struct Game;

struct GameIterator
{
    const Game &game;
    int index{0};
    using iterator_category = std::forward_iterator_tag;
    using difference_type = size_t;
    using value_type = Position;
    using pointer = Position *;   // or also value_type*
    using reference = Position &; // or also value_type&

    GameIterator(const GameIterator &other) : game(other.game)
    {
        index = other.index;
    }

    GameIterator(const Game &game) : game(game)
    {
    }

    bool operator==(GameIterator &other) const;

    bool operator!=(GameIterator &other) const;

    GameIterator &operator++()
    {
        index++;
        return *this;
    }

    GameIterator &operator--()
    {
        index--;
        return *this;
    }

    GameIterator operator++(int)
    {
        GameIterator copy(*this);
        index++;
        return copy;
    }

    GameIterator operator--(int)
    {
        GameIterator copy(*this);
        index--;
        return copy;
    }

    Position operator*() const;
};

struct Game
{
    Position start_position;
    std::vector<uint8_t> indices;
    Result result{Result::UNKNOWN};

    Game(const Position start_position) : start_position(start_position)
    {
    }

    Game() = default;

    void add_move(uint8_t move_index)
    {
        uint8_t encoding = 0;
        Game::encode_move(encoding, move_index);
    }

    void set_result(Result res)
    {
        result = res;
    }

    void set_result(Result res, int n)
    {
        Game::encode_result(indices[n], res);
    }

    void add_position(Position pos)
    {
        if (start_position.is_empty())
        {
            start_position = pos;
            return;
        }

        MoveListe liste;
        Position x = get_last_position();
        get_moves(x, liste);
        for (auto i = 0; i < liste.length(); ++i)
        {
            Position t = x;
            t.make_move(liste[i]);
            if (t == pos)
            {
                uint8_t encoding = 0;
                Game::encode_move(encoding, i);
                indices.emplace_back(encoding);
                break;
            }
        }
    }

    // overloading read write operators

    friend std::istream &operator>>(std::istream &stream, Game &game)
    {
        stream >> game.start_position;
        uint16_t num_indices;
        stream.read((char *)&num_indices, sizeof(uint16_t));
        stream.read((char *)&game.result, sizeof(Result));
        game.indices = std::vector<uint8_t>(num_indices);
        stream.read((char *)&game.indices[0], sizeof(uint8_t) * num_indices);
        return stream;
    }

    friend std::ostream &operator<<(std::ostream &stream, const Game &game)
    {
        Position start_pos = game.start_position;
        stream << start_pos;
        const uint16_t num_indices = game.indices.size();
        stream.write((char *)&num_indices, sizeof(uint16_t));
        stream.write((char *)&game.result, sizeof(Result));
        stream.write((char *)&game.indices[0], sizeof(uint8_t) * num_indices);
        return stream;
    }

    Position get_position(int n) const
    {
        // returns the position after the nth move
        Position current = start_position;
        for (auto i = 0; i < n; ++i)
        {
            MoveListe liste;
            get_moves(current, liste);
            current.make_move(liste[Game::get_move_index(indices[i])]);
        }
        return current;
    }

    Position get_last_position() const
    {
        return get_position(indices.size());
    }

    bool operator==(Game &other) const
    {
        return (other.start_position == start_position &&
                std::equal(indices.begin(), indices.end(), other.indices.begin()));
    }

    bool operator!=(Game &other) const
    {
        return !((*this) == other);
    }

    GameIterator begin() const
    {
        GameIterator beg(*this);
        return beg;
    }

    GameIterator end() const
    {
        GameIterator beg(*this);
        beg.index = indices.size() + 1;
        return beg;
    }

    static void encode_move(uint8_t &bit_field, uint8_t move_index)
    {
        // clearing the move field
        bit_field &= ~(63);
        bit_field |= move_index;
    }

    static void encode_result(uint8_t &bit_field, Result result)
    {
        uint8_t temp = static_cast<uint8_t>(result);
        const uint8_t clear_bits = 3ull << 6;
        bit_field &= ~clear_bits;
        bit_field |= temp << 6;
    }

    static uint8_t get_move_index(const uint8_t &bit_field)
    {
        const uint8_t clear_bits = 3ull << 6;
        uint8_t copy = bit_field & (~clear_bits);
        return copy;
    }

    static Result get_result(uint8_t &bit_field)
    {
        uint8_t copy = (bit_field >> 6) & 3ull;
        return static_cast<Result>(copy);
    }

    template <typename OutIter, typename Lambda>
    void extract_samples_test(OutIter iterator, Lambda lambda)
    {
        // to be checked and continued
        Position current = start_position;

        if (current.is_empty())
            return;

        for (auto i = 0; i < indices.size(); ++i)
        {
            Sample sample;
            MoveListe liste;
            get_moves(current, liste);
            sample.position = current;
            sample.result = Game::get_result(indices[i]);
            Move m = liste[Game::get_move_index(indices[i])];
            sample.move = lambda(current.get_color(), m);

            current.make_move(m);
            *iterator = sample;
            iterator++;
        }
        Sample sample;
        sample.position = current;
        sample.result = result;
        sample.move = -1;
        *iterator = sample;
    }

    template <typename OutIter>
    void extract_samples_test(OutIter iterator)
    {
        auto lambda = [](Color color, Move move)
        {
            return Statistics::mPicker.get_move_encoding(color, move);
        };
        return extract_samples_test(iterator, lambda);
    }
};

inline Position GameIterator::operator*() const
{
    Position current = game.get_position(index);
    return current;
}

inline bool GameIterator::operator==(GameIterator &other) const
{
    return (other.index == index);
}

inline bool GameIterator::operator!=(GameIterator &other) const
{
    return (other.index != index);
}

template <typename Iterator>
inline void merge_training_data(Iterator begin, Iterator end, std::string output)
{
    std::ofstream stream_out(output, std::ios::app | std::ios::binary);
    std::cout << "Merging files" << std::endl;
    if (!stream_out.good())
    {
        std::cerr << "Could not merge files" << std::endl;
        std::exit(-1);
    }
    for (auto it = begin; it != end; ++it)
    {
        auto file_input = *it;
        std::ifstream stream(file_input.c_str());
        if (!stream.good())
        {
            std::cerr << "Could not open the stream" << std::endl;
            std::exit(-1);
        }
        Game game;
        while (stream >> game)
        {
            stream_out << game;
        }
    }
    // this should merge alle the files into one
}

inline std::optional<std::string> is_temporary_train_file(std::string name);

template <typename Iterator>
inline std::pair<size_t, size_t> count_unique_positions(Iterator begin, Iterator end)
{
    BloomFilter<Position> filter(9585058378, 3);
    size_t unique_count = 0;
    size_t total_positions = 0;
    std::for_each(begin, end, [&](Game game)
                  {
         for (auto pos: game) {
            if (!filter.has(pos)) {
                unique_count++;
                filter.insert(pos);
            }
            total_positions++;
        }
        return; });
    return std::make_pair(unique_count, total_positions);
}

std::pair<size_t, size_t> count_unique_positions(std::string game_file);


inline size_t count_trainable_positions(std::string game_file, std::pair<size_t, size_t> range){
    std::ifstream stream(game_file, std::ios::binary);
    std::istream_iterator<Game> begin(stream);
    std::istream_iterator<Game> end;
    size_t counter{0};
    // temporary before I can speed this thing up
    // way too slow
    std::for_each(begin, end, [&](const Game &g)
                  { counter += g.indices.size() + 1; });
    return counter;

}
void merge_temporary_files(std::string directory, std::string out_directory);

void create_subset(std::string file, std::string output, size_t num_games);

auto get_piece_distrib(std::ifstream &stream);

auto get_piece_distrib(std::string input);

auto get_capture_distrib(std::ifstream &stream);
auto get_capture_distrib(std::string input);
#endif // READING_COMPRESS_H
