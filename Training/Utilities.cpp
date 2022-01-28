//
// Created by robin on 5/21/18.
//

#include "Utilities.h"


namespace Utilities {

    std::unordered_set<uint64_t> hashes;


    void create_samples_from_games(std::string games, std::string output) {
        size_t uniq_count{0};
        size_t total_count{0};
        BloomFilter<Sample> filter(5751035027, 10);
        std::vector<Sample> buffer;
        const size_t max_cap_buffer = 10000;

        size_t game_counter = 0;
        //first stepping through the games
        std::ifstream stream(games, std::ios::binary);
        std::ofstream out_stream(output, std::ios::binary);

        std::istream_iterator<Position> end;

        std::vector<Position> game;
        auto it = std::istream_iterator<Position>(stream);
        Position previous = *it;
        game.emplace_back(previous);
        ++it;
        for (; it != end; ++it) {
            Position pos = *it;
            const size_t piece_count = Bits::pop_count(pos.BP | pos.WP);
            const size_t prev_piec_count = Bits::pop_count(previous.BP | previous.WP);
            if (piece_count <= prev_piec_count) {
                game.emplace_back(pos);
            } else {
                //getting the move

                //some form of sanity check

                Position &back = game.back();
                const size_t rep_count = std::count(game.begin(), game.end(), back);

                MoveListe liste;
                get_moves(back, liste);
                if (liste.length() == 0) {
                    const int result = -back.color;

                    for (auto p: game) {
                        Sample s;
                        s.position = p;
                        if(result ==-1)
                            s.result = BLACK_WON;
                        else if(result ==1)
                            s.result = WHITE_WON;
                        else
                            s.result = DRAW;
                        if (!filter.has(s)) {
                            filter.insert(s);
                            buffer.emplace_back(s);
                            uniq_count++;
                        }
                    }
                } else if (rep_count >= 3) {


                    for (auto p: game) {
                        Sample s;
                        s.position = p;
                        s.result = DRAW;
                        if (!filter.has(s)) {
                            filter.insert(s);
                            buffer.emplace_back(s);
                            uniq_count++;
                        }
                    }
                }


                game.clear();
                game_counter++;
            }
            previous = pos;
            if (buffer.size() >= max_cap_buffer) {
                std::copy(buffer.begin(), buffer.end(), std::ostream_iterator<Sample>(out_stream));
                buffer.clear();
            }
            total_count++;
        }
        std::cout << "Total Position: " << total_count << " after removing: " << uniq_count << std::endl;
        std::copy(buffer.begin(), buffer.end(), std::ostream_iterator<Sample>(out_stream));
    }
}
