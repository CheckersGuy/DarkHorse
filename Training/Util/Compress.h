//
// Created by robin on 21.03.22.
//

#ifndef READING_COMPRESS_H
#define READING_COMPRESS_H
#include <assert.h>
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



struct __attribute__((packed)) Encoding {
    uint32_t move_index :6 =50;
    uint32_t result : 2 =0;

    bool operator ==(const Encoding& other)const {
        return move_index ==other.move_index && result == other.result;
    };

    bool operator !=(const Encoding& other)const {
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
            assert(move_index<liste.length());
           Move move = liste[move_index];
		   assert(!move.is_empty());
            current.make_move(move);
        }
        return current;
    }

    Position get_last_position() const
    {
        return get_position(indices.size());
    }

    auto get_game_result() {
        Position last_position = get_last_position();
        MoveListe liste;
        get_moves(last_position,liste);
        if(liste.length()==0 && last_position.get_color()==BLACK) {
            return WHITE_WON;
        } else if(liste.length()==0 && last_position.get_color()==WHITE) {
            return BLACK_WON;
        }
        //questions: Should positions that are not valid for rescoring but
        //end in 3 fold repetitions be ignored ?, sounds interesting to me and I will try it

        return UNKNOWN;

    }

    template<typename Iterator> void extract_positions(Iterator iter) {
        Position current =start_position;
        *iter =start_position;
        iter++;
        for(auto i=0; i<indices.size(); ++i) {
            MoveListe liste;
            get_moves(current,liste);
            Move move =liste[indices[i].move_index];
            current.make_move(move);
            *iter=current;
            iter++;
        }
    }

    template<typename Oracle> void rescore_game(Oracle func) {
        //Oracle provides true-results for valid positions:
        Position current = start_position;
        auto end_result = get_game_result();
        int last_stop=-1;
        for(auto i=0; i<indices.size(); ++i) {
            auto result = func(current);
            if(result == UNKNOWN)
                continue;
            for(int k=i; k>last_stop; k--) {
                indices[k].result = result;
            }
            last_stop = i;
        }
        for(auto& enc : indices) {
            if(enc.result!=UNKNOWN)
                continue;
            enc.result = end_result;
        }
    }

    bool operator==(const Game& other) const
    {
        return (other.start_position == start_position &&
                std::equal(indices.begin(), indices.end(), other.indices.begin()));
    }

    bool operator!=(const Game& other) const
    {
        return !(other.start_position == start_position &&
                 std::equal(indices.begin(), indices.end(), other.indices.begin()));
    }



    template <typename OutIter>
    void extract_samples(OutIter iterator)
    {
        // to be checked and continued
        Position current = start_position;


        for (auto i = 0; i < indices.size(); ++i)
        {

            auto encoding = indices[i];
            Sample sample;
            MoveListe liste;
            get_moves(current, liste);
            sample.position = current;
            sample.result = static_cast<Result>(encoding.result);
            Move m = liste[encoding.move_index];
            sample.move = Statistics::mPicker.get_move_encoding(current.get_color(), m);

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
};



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
    std::vector<Position>positions;
    positions.reserve(600);
    std::for_each(begin, end, [&](Game game)
    {   positions.clear();
        game.extract_positions(std::back_inserter(positions));
        for (auto pos: positions) {
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
