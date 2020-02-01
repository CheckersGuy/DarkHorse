//
// Created by robin on 7/26/18.
//

#ifndef TRAINING_MATCH_H
#define TRAINING_MATCH_H

#include <deque>
#include  <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include  <sys/types.h>
#include "fcntl.h"
#include "Utilities.h"


inline std::string getPositionString(Position pos) {
    std::string position;
    for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t current = 1u << i;
        if ((current & (pos.BP & pos.K))) {
            position += "3";
        } else if ((current & (pos.WP & pos.K))) {
            position += "4";
        } else if ((current & pos.BP)) {
            position += "1";
        } else if ((current & pos.WP)) {
            position += "2";
        } else {
            position += "0";
        }
    }
    if (pos.getColor() == BLACK) {
        position += "B";
    } else {
        position += "W";
    }
    return position;
}

struct Engine {
    enum class State {
        Idle, Game_Ready, Init_Ready,
        Update
    };
    State state;
    const int &engineRead;
    const int &engineWrite;
    bool waiting_response = false;
    int time_move = 100;
    int hash_size = 21;

    void initEngine();

    void writeMessage(const std::string &msg);

    void newGame(const Position &pos);

    void update();

    std::optional<Move> search();

    std::string readPipe();

    void setHashSize(int hash);

    void setTime(int time);
};


struct Interface {

    std::array<Engine, 2> engines;
    Position pos;
    int first_mover = 0;
    std::vector<Position> history;

    void process();

    bool is_n_fold(int n);

    bool isLegalMove(Move move);

    bool is_terminal_state();

    void reset_engines();
};


class Match {

private:
    const std::string &first;
    const std::string &second;
    int time;
    int hash_size;
    int maxGames;
    int wins_one{0}, wins_two{0}, draws{0};
    int threads;
    std::string openingBook;
    const std::string log_file;
    std::ofstream logger;

public:
    Match(const std::string &first, const std::string &second) : first(first), second(second), draws(0), maxGames(1000),
                                                                 time(100), threads(1), openingBook(
                    "/home/robin/DarkHorse/Training/Positions/genBook6.pos"), log_file("log.txt") {
        logger = std::ofstream(log_file, std::ios::app);
        if (!logger.good()) {
            std::cerr << "Error log_file" << std::endl;
            exit(EXIT_FAILURE);
        }
    };

    ~Match();

    void setMaxGames(int games);

    int getMaxGames();

    void start();

    void setHashSize(int hash);

    void setTime(int time);

    void setNumThreads(int threads);

    int getNumThreads();

    std::string &getOpeningBook();

    void setOpeningBook(std::string book);


};


#endif //TRAINING_MATCH_H
