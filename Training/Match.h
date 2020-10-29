//
// Created by robin on 7/26/18.
//

#ifndef TRAINING_MATCH_H
#define TRAINING_MATCH_H

#include <deque>
#include  <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "fcntl.h"
#include "Utilities.h"
#include <sys/resource.h>
#include "proto/Training.pb.h"
#include <filesystem>
#include <HyperLog.h>

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

struct TrainHasher{

    uint64_t operator()(Position p){
        return p.key;
    }
};

class Logger {

private:

    std::ofstream stream;
    const std::string log_file = "log.txt";
    bool logging{false};
    Logger() {
        stream = std::ofstream(log_file);
        if (!stream.good()) {
            std::cerr << "Couldnt initialize logger" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    ~Logger() {
        stream.close();
    }

public:

    void turn_on() {
        logging = true;
    }

    void turn_off() {
        logging = false;
    }

    template<typename T>
    void write_log_message(T arg) {
        if (!logging)
            return;
        stream << arg;
    }


    template<typename T>
    Logger &operator<<(T msg) {
        write_log_message(msg);
        return *this;
    }


    static Logger &get_instance() {
        static Logger logger;
        return logger;
    }

};

struct Engine {
    enum class State {
        Idle, Game_Ready, Init_Ready,
        Update
    };
    std::string engine_name;
    State state;
    const int &engineRead;
    const int &engineWrite;
    bool waiting_response = false;
    int time_move = 100;
    int hash_size = 21;

    void initEngine();

    void writeMessage(const std::string &msg);

    void new_move(Move move);

    void newGame(const Position &pos);

    void update();

    Move search();

    std::string readPipe();

    void setHashSize(int hash);

    void setTime(int time);
};


struct Interface {

    std::array<Engine, 2> engines;
    Position pos;
    int first_mover = 0;
    std::vector<Position> history;
    bool played_reverse = false;

    void process();

    bool is_n_fold(int n);

    bool is_legal_move(Move move);

    bool is_terminal_state();

    void reset_engines();

    void terminate_engines();
};


class Match {

private:
    const std::string &first;
    const std::string &second;
    int time{100};
    int hash_size{21};
    int maxGames{1000};
    int wins_one{0}, wins_two{0}, draws{0};
    int threads{1};
    bool play_reverse{false};
    std::string openingBook{"../Training/Positions/3move.book"};
    std::string output_file;
    Training::TrainData data;
    void addPosition(Position pos, Training::Result result);

public:
    explicit Match(const std::string &first, const std::string &second, std::string output) : first(
            first), output_file(output),
                                                                                              second(second) {
        std::ifstream stream(output.c_str(), std::ios::binary);
        if (stream.good()) {
            data.ParseFromIstream(&stream);
        }
        stream.close();
    };

    void setMaxGames(int games);

    int getMaxGames();

    void start();

    void setHashSize(int hash);

    void setTime(int time);

    void setNumThreads(int threads);

    void set_play_reverse(bool flag);

    std::string get_output_file();

    int getNumThreads();

    std::string &getOpeningBook();

    void setOpeningBook(std::string book);


};


#endif //TRAINING_MATCH_H
