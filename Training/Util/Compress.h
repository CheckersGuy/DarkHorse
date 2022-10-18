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
    using iterator_category = std::forward_iterator_tag;
    using difference_type = size_t;
    using value_type = Position;
    using pointer = Position *;   // or also value_type*
    using reference = Position &; // or also value_type&
    const Game &game;
    int index{0};
    Position current;
    GameIterator(const GameIterator &other);
    GameIterator(const Game &game);
    bool operator==(GameIterator other) const;

    bool operator!=(GameIterator other) const;

    GameIterator &operator++();
    GameIterator operator++(int);
    Position operator*() const;
};


struct __attribute__((packed)) Encoding {
    uint32_t move_index :6 =50;
    uint32_t result : 2 =0;

    bool operator ==(Encoding other) {
        return move_index ==other.move_index && result == other.result;
    };

    bool operator !=(Encoding other) {
        return move_index !=other.move_index || result !=other.result;
    };
};

struct Game
{
    Position start_position;
    std::vector<Encoding> indices;
    Result result{Result::UNKNOWN};

    Game(const Position start_position) : start_position(start_position)
    {
    }

    Game() = default;

    void add_move(uint32_t move_index)
    {

    }

    void set_result(Result res)
    {
        result = res;
    }

    void set_result(Result res, int n)
    {
        indices[n].result =static_cast<int>(res);
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
                Encoding encoding;
                encoding.move_index = i;
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
        game.indices =std::vector<Encoding>(num_indices);
        stream.read((char *)&game.indices[0], sizeof(Encoding) * num_indices);
        return stream;
    }

    friend std::ostream &operator<<(std::ostream &stream, const Game &game)
    {
        Position start_pos = game.start_position;
        stream << start_pos;
        const uint16_t num_indices = game.indices.size();
        stream.write((char *)&num_indices, sizeof(uint16_t));
        stream.write((char *)&game.result, sizeof(Result));
        stream.write((char *)&game.indices[0], sizeof(Encoding) * num_indices);
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
            auto move_index = indices[i].move_index;
            current.make_move(liste[move_index]);
        }
        return current;
    }

    Position get_last_position() const
    {
        return get_position(indices.size());
    }

    bool operator==(Game other) const
    {
        return (other.start_position == start_position &&
                std::equal(indices.begin(), indices.end(), other.indices.begin()));
    }

    bool operator!=(Game other) const
    {
        return !(other.start_position == start_position &&
                 std::equal(indices.begin(), indices.end(), other.indices.begin()));
    }

    GameIterator begin() const
    {
        GameIterator beg(*this);
        beg.current = start_position;
        return beg;
    }

    GameIterator end() const
    {
        GameIterator beg(*this);
        beg.index = indices.size() + 1;
        return beg;
    }

//    static void encode_move(uint8_t &bit_field, uint8_t move_index)
//    {
//        // clearing the move field
//        bit_field &= ~(63);
//        bit_field |= move_index;
//    }
//
//    static void encode_result(uint8_t &bit_field, Result result)
//    {
//        uint8_t temp = static_cast<uint8_t>(result);
//        const uint8_t clear_bits = 3ull << 6;
//        bit_field &= ~clear_bits;
//        bit_field |= temp << 6;
//    }
//
//    static uint8_t get_move_index(const uint8_t &bit_field)
//    {
//        const uint8_t clear_bits = 3ull << 6;
//        uint8_t copy = bit_field & (~clear_bits);
//        return copy;
//    }
//
//    static Result get_result(uint8_t &bit_field)
//    {
//        uint8_t copy = (bit_field >> 6) & 3ull;
//        return static_cast<Result>(copy);
//    }

    template <typename OutIter, typename Lambda>
    void extract_samples_test(OutIter iterator, Lambda lambda)
    {
        // to be checked and continued
        Position current = start_position;

        if (current.is_empty())
            return;

        for (auto i = 0; i < indices.size(); ++i)
        {

            auto encoding = indices[i];
            Sample sample;
            MoveListe liste;
            get_moves(current, liste);
            sample.position = current;
            sample.result = static_cast<Result>(encoding.result);
            Move m = liste[encoding.move_index];
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


inline GameIterator::GameIterator(const GameIterator &other) : game(other.game)
    {
        index = other.index;
        current =other.current;
    }

inline GameIterator::GameIterator(const Game &game) : game(game)
    {
			current = game.start_position;
    }

    inline GameIterator & GameIterator::operator++()
    {
		MoveListe liste;
		get_moves(current,liste);
		auto move_index = game.indices[index].move_index;
		current.make_move(liste[move_index]);
        index++;
        return *this;
    }


    inline GameIterator GameIterator::operator++(int)
    {
        GameIterator copy(*this);
        index++;
        return copy;
    }


inline Position GameIterator::operator*() const
{
    return current;
}

inline bool GameIterator::operator==(GameIterator other) const
{
    return (other.game == game &&  other.index == index);
}

inline bool GameIterator::operator!=(GameIterator other) const
{

    return !(other.game == game &&  other.index == index);
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
        return;
    });
    return std::make_pair(unique_count, total_positions);
}

std::pair<size_t, size_t> count_unique_positions(std::string game_file);


inline size_t count_trainable_positions(std::string game_file) {
    std::ifstream stream(game_file, std::ios::binary);
    std::istream_iterator<Game> begin(stream);
    std::istream_iterator<Game> end;
    size_t counter{0};
    // temporary before I can speed this thing up
    // way too slow
    std::for_each(begin, end, [&](const Game &g)
    {
        counter += g.indices.size() + 1;
    });
    return counter;

}
void merge_temporary_files(std::string directory, std::string out_directory);

void create_subset(std::string file, std::string output, size_t num_games);

auto get_piece_distrib(std::ifstream &stream);

auto get_piece_distrib(std::string input);

auto get_capture_distrib(std::ifstream &stream);
auto get_capture_distrib(std::string input);
#endif // READING_COMPRESS_H
