//
// Created by robin on 5/21/18.
//

#ifndef TRAINING_UTILITIES_H
#define TRAINING_UTILITIES_H

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <filesystem>

#include <boost/iostreams/copy.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include "Utilities.h"

#include <fstream>
#include "Board.h"
#include "GameLogic.h"
#include "MGenerator.h"
#include <boost/interprocess/streams/bufferstream.hpp>
#include <unordered_set>
#include <iterator>
#include <future>
#include <algorithm>

namespace Utilities {

    template<typename Iter, typename Function>
    void parallel_for(Iter begin, Iter end, Function func, size_t num_threads = std::thread::hardware_concurrency()) {
        struct thread_waiter {
            std::vector<std::thread> &threads;


            ~thread_waiter() {
                for (auto &th : threads)
                    th.join();
            }
        };
        num_threads = num_threads - 1u;

        auto data_size = std::distance(begin, end);
        if (data_size == 0)
            return;
        auto block_size = data_size / (num_threads + 1);
        num_threads = (block_size == 0) ? data_size - 1 : num_threads;
        block_size = (block_size == 0) ? 1 : block_size;
        std::vector<std::thread> threads;
        std::vector<std::future<void>> futures;
        thread_waiter waiter{threads};
        auto begin_block = begin;
        for (auto i = 0; i < num_threads; ++i) {
            auto end_block = begin_block;
            std::advance(end_block, block_size);

            std::packaged_task<void()> task([=]() {
                std::for_each(begin_block, end_block, func);
            });
            futures.emplace_back(task.get_future());
            threads.emplace_back(std::thread(std::move(task)));
            begin_block = end_block;
        }
        std::for_each(begin_block, end, func);
        for (auto &fu : futures) {
            fu.get();
        }
    }


    void createNMoveBook(std::vector<Position> &data, int N, Board &board, Value lowerBound, Value upperBound);

    void createNMoveBook(std::vector<Position> &pos, int N, Board &board);

    void loadPositions(std::vector<Position> &positions, const std::string file);

    void savePositions(std::vector<Position> &positions, const std::string file);

}


#endif //TRAINING_UTILITIES_H
