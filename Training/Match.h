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
        if ((current & pos.BP)) {
            position += "1";
        } else if ((current & pos.WP)) {
            position += "2";
        } else if ((current & (pos.BP & pos.K))) {
            position += "3";
        } else if ((current & (pos.WP & pos.K))) {
            position += "4";
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

inline Position posFromString(const std::string &pos) {
    Position result;
    for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t current = 1u << i;
        if (pos[i] == '1') {
            result.BP |= current;
        } else if (pos[i] == '2') {
            result.WP |= current;
        } else if (pos[i] == '3') {
            result.K |= current;
            result.BP |= current;
        } else if (pos[i] == '4') {
            result.K |= current;
            result.WP |= current;
        }
    }
    if (pos[32] == 'B') {
        result.color = BLACK;
    } else {
        result.color = WHITE;
    }
    result.key = Zobrist::generateKey(result);
    return result;
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
};


class Match {

private:
    const std::string &first;
    const std::string &second;
    int time;
    int maxGames;
    int wins, losses, draws;
    int threads;
    std::string openingBook;

public:

    Match() = default;

    Match(const std::string &first, const std::string &second) : first(first), second(second), wins(0), losses(0), draws(0), maxGames(1000),
                                           time(100), threads(1), openingBook("Positions/3move.pos") {};

    void setMaxGames(int games);

    int getMaxGames();

    void start();

    int getWins();

    int getLosses();

    int getDraws();

    int getElo();

    void setTime(int time);

    void setNumThreads(int threads);

    int getNumThreads();

    std::string &getOpeningBook();

    void setOpeningBook(std::string book);


};


#endif //TRAINING_MATCH_H
