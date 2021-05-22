//
// Created by root on 19.05.21.
//

#ifndef READING_GENERATORZ_H
#define READING_GENERATORZ_H


#include <cstdint>
#include <types.h>
#include <Position.h>
#include <condition_variable>
#include <Board.h>
#include <GameLogic.h>

struct TrainSample {
    int result;
    int evaluation;
    Position pos;

    friend std::ostream &operator<<(std::ostream &stream, const TrainSample& s);

    friend std::istream &operator>>(std::istream& stream, TrainSample &s);

};

class GeneratorZ {

private:
    std::condition_variable cond_var;
    bool stop{false};
    uint64_t max_time;
    uint64_t max_depth{MAX_PLY};
    uint64_t max_games{0};
    uint64_t op_index{0};
    size_t num_threads{1};
    std::vector<Position> openings;
    std::vector<TrainSample> samples;
    std::vector<std::jthread> threads;
    std::mutex my_mutex;
    std::atomic<uint64_t> num_games{0};

public:

    void start_threads();

    void generate_games();

    template<typename Iterator>
    void push_game(Iterator begin, Iterator end) {
        std::lock_guard guard(my_mutex);
        std::copy(begin, end, std::back_inserter(samples));
    }


};


#endif //READING_GENERATORZ_H
