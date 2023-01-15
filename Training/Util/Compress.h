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
#include "MovePicker.h"
#include <map>
#include <vector>

struct Game;



struct Encoding {
    uint8_t encoding{0};

    Encoding(){
        set_move_index(50);
    }

    void set_move_index(int move_index){
        //clearing the bits
        encoding&=~63;
        encoding |= static_cast<uint8_t>(move_index);
    }

    void set_result(Result res){
        encoding&=~192;
        const uint8_t value =static_cast<uint8_t>(res);
        encoding|=value<<6;
    }

    Result get_result()const{
        return static_cast<Result>(encoding>>6);
    }

    int get_move_index()const{
        return static_cast<uint8_t>(encoding&63);
    }

    bool operator==(const Encoding& other)const{
        return other.encoding == encoding;
    }

     bool operator!=(const Encoding& other)const{
        return other.encoding != encoding;
    }

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
        indices[n].set_result(res);
    }

    bool add_position(Position pos)
    {
        //returns false if the operation failed
        if (start_position.is_empty())
        {
            start_position = pos;
            return true;
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
                encoding.set_move_index(i);
                indices.emplace_back(encoding);
                return true;
            }
        }
        return false;
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
            auto move_index = indices[i].get_move_index();
            Move move = liste[move_index];
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

        std::vector<Position>positions;
        extract_positions(std::back_inserter(positions));
        auto count = std::count(positions.begin(),positions.end(),last_position);
        if(count>=3) {
            return DRAW;
        }

        return UNKNOWN;

    }

    template<typename Iterator> void extract_positions(Iterator iter) {
        Position current =start_position;
        *iter =start_position;
        iter++;
        for(auto i=0; i<indices.size(); ++i) {
            MoveListe liste;
            get_moves(current,liste);
            Move move =liste[indices[i].get_move_index()];
            current.make_move(move);
            *iter=current;
            iter++;
        }
    }

    template<typename Oracle> void rescore_game(Oracle func) {
        //Oracle provides true-results for valid positions:
        Position current = start_position;
        const auto end_result = get_game_result();
        result = end_result;
        int last_stop=-1;
        for(auto i=0; i<indices.size(); ++i) {
            auto o_result = func(current);
            MoveListe liste;
            get_moves(current,liste);
            current.make_move(liste[indices[i].get_move_index()]);
            indices[i].set_result(result);
            if(o_result == UNKNOWN)
                continue;
            for(int k=i; k>last_stop; k--) {
                indices[k].set_result(o_result);
            }
            last_stop = i;
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
            sample.result = static_cast<Result>(encoding.get_result());
            Move m = liste[encoding.get_move_index()];
            int move_encoding;
            if(sample.position.get_color() == BLACK){
              Move temp;
              temp.from = getMirrored(m.from);
              temp.to = getMirrored(m.to);
              move_encoding = Statistics::MovePicker::get_move_encoding(temp);
            }else{
              move_encoding = Statistics::MovePicker::get_move_encoding(m);
            }
            current.make_move(m);
            sample.move = move_encoding;
            if(sample.position.has_jumps() || sample.result==UNKNOWN || sample.position.piece_count()<=10)
                continue;
            *iterator = sample;
            iterator++;
        }
        Sample sample;
        sample.position = current;
        sample.result = result;
        sample.move = -1;
        if(sample.position.has_jumps() || sample.result==UNKNOWN)
                return;
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
