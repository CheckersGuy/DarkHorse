//
// Created by robin on 5/21/18.
//

#include "Utilities.h"


namespace Utilities {

    std::unordered_set<uint64_t> hashes;


    void create_samples_from_games(std::string games, std::string output) {
        size_t uniq_count{0};
        size_t total_count{0};
        SampleFilter filter(5751035027, 10);
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
                for (auto c: game) {
                    c.printPosition();
                    std::cout << "Fen: " << c.get_fen_string() << std::endl;
                    std::cout << std::endl;
                }
                Position &back = game.back();
                const size_t rep_count = std::count(game.begin(), game.end(), back);

                MoveListe liste;
                getMoves(back, liste);
                if (liste.length() == 0) {
                    const int result = -back.color;
                    std::cout<<((result ==-1) ? "Black won": "White won");
                    for (auto p: game) {
                        Sample s;
                        s.position = p;
                        s.result = result;
                        if (!filter.has(s)) {
                            filter.insert(s);
                            buffer.emplace_back(s);
                            uniq_count++;
                        }
                    }
                } else if (rep_count >= 3) {
                    std::cout << "Repetition" << std::endl;
                    std::cout << "Fen: " << back.get_fen_string() << std::endl;

                    for (auto p: game) {
                        Sample s;
                        s.position = p;
                        s.result = 0;
                        if (!filter.has(s)) {
                            filter.insert(s);
                            buffer.emplace_back(s);
                            uniq_count++;
                        }
                    }
                }
                for (auto i = 0; i < 3; ++i)
                    std::cout << "\n";


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
        std::cout<<"Total Position: "<<total_count<<" after removing: "<<uniq_count<<std::endl;
        std::copy(buffer.begin(), buffer.end(), std::ostream_iterator<Sample>(out_stream));
    }
}
