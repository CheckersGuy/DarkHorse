


#include <vector>
#include <future>
#include <bitset>
#include <map>
#include "Transposition.h"
#include "GameLogic.h"
#include <list>
#include <iterator>
#include <algorithm>
#include "Perft.h"
#include  <sys/types.h>
#include <sys/wait.h>
#include  <sys/types.h>
#include <unistd.h>
#include "fcntl.h"
#include <regex>

std::optional<Move> decodeMove(const std::string &move_string) {
    std::regex reg("[0-9]{1,2}[|][0-9]{1,2}([|][0-9]{1,2})*");
    if (move_string.size() > 2)
        if (std::regex_match(move_string, reg)) {
            Move result;
            std::regex reg2("[^|]+");
            std::sregex_iterator iterator(move_string.begin(), move_string.end(), reg2);
            auto from = (*iterator++).str();
            auto to = (*iterator++).str();
            result.from = 1u << std::stoi(from);
            result.to = 1u << std::stoi(to);
            for (auto it = iterator; it != std::sregex_iterator{}; ++it) {
                auto value = (*it).str();
                result.captures |= 1u << std::stoi(value);
            }
            return std::make_optional(result);
        }
    return std::nullopt;
}

std::string encodeMove(Move move) {
    std::string move_string;
    move_string += std::to_string(move.getFromIndex());
    move_string += "|";
    move_string += std::to_string(move.getToIndex());
    if (move.captures)
        move_string += "|";

    uint32_t lastMove = (move.captures == 0u) ? 0u : Bits::bitscan_foward(move.captures);
    uint32_t temp = move.captures & (~(1u << lastMove));
    while (temp) {
        uint32_t mSquare = Bits::bitscan_foward(temp);
        temp &= temp - 1u;
        move_string += std::to_string(mSquare);
        move_string += "|";
    }
    if (lastMove) {
        move_string += std::to_string(lastMove);
    }
    return move_string;
}


std::string getPositionString(Position pos) {
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

Position posFromString(const std::string &pos) {
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
    int hash_size=21;

    void initEngine();

    void writeMessage(const std::string &msg);

    void newGame(const Position &pos);

    void update();

    std::optional<Move> search();

    std::string readPipe();

    void setHashSize(int hash);

    void setTime(int time);
};

void Engine::setTime(int time) {
    time_move = time;
}

void Engine::setHashSize(int hash) {
    hash_size = hash;
}


std::string Engine::readPipe() {
    std::string message;
    char c;
    int result;
    do {
        result = read(engineRead, (char *) (&c), sizeof(char));
        if (c == '\n') {
            break;
        } else {
            message += c;
        }

    } while (result != -1);
    return message;
}

void Engine::writeMessage(const std::string &msg) {
    const std::string ex_message = msg + "\n";
    write(engineWrite, (char *) &ex_message.front(), sizeof(char) * ex_message.size());
}

std::optional<Move> Engine::search() {
    if (state == State::Game_Ready) {
        if (!waiting_response) {
            writeMessage("search");
            writeMessage(std::to_string(time_move));
        }
        waiting_response = true;
        auto answer = readPipe();
        auto move = decodeMove(answer);
        if (move.has_value()) {
            waiting_response = false;
            return move;
        }
    }
    return std::nullopt;
}


void Engine::newGame(const Position &pos) {
    if (state == State::Init_Ready) {
        if (!waiting_response) {
            writeMessage("new_game");
            writeMessage(getPositionString(pos));
        }
        waiting_response = true;
        auto answer = readPipe();
        if (answer == "game_ready") {
            state = State::Game_Ready;
            waiting_response = false;
            std::cout << "Game ready" << std::endl;
        }
    }
}

void Engine::update() {
    if (state == State::Update) {
        auto answer = readPipe();
        if (answer == "update_ready") {
            state = State::Game_Ready;
            std::cout << "Updated" << std::endl;
        }
    }
}

void Engine::initEngine() {
    if (state == State::Idle) {
        if (!waiting_response) {
            writeMessage("init");
            writeMessage(std::to_string(hash_size));
        }
        waiting_response = true;
        auto answer = readPipe();
        if (answer == "init_ready") {
            state = State::Init_Ready;
            waiting_response = false;
            std::cout << "Init ready" << std::endl;
        }
    }

}


struct Interface {

    std::array<Engine, 2> engines;
    Board board;
    int first_mover = 0;
    void process();
};

void Interface::process() {
    MoveListe liste;
    getMoves(board.getPosition(), liste);
    if (liste.length() == 0 || board.isRepetition()) {
        return;
    }
    for (auto &engine : engines) {
        engine.initEngine();
        engine.newGame(board.getPosition());
        engine.update();
    }
    const int second_mover = (first_mover == 0) ? 1 : 0;
    auto move = engines[first_mover].search();
    if (move.has_value()) {
        board.makeMove(move.value());
        engines[second_mover].state = Engine::State::Update;
        engines[second_mover].writeMessage("update");
        engines[second_mover].writeMessage(encodeMove(move.value()));
        first_mover = second_mover;
        board.printBoard();
        std::cout << std::endl;
    }


}


int main(int argl, const char **argc) {
/*     std::string current;
     Board board;
     board = Position::getStartPosition();
     while (std::cin >> current) {
         if (current == "init") {
             initialize();
             std::string hash_string;
             std::cin>>hash_string;
             const int hash_size = std::stoi(hash_string);
             setHashSize(1u<<hash_size);
             std::cerr<<"HashSize: "<<hash_string<<std::endl;
             std::cout << "init_ready" << "\n";
         } else if (current == "new_game") {
             std::string position;
             std::cin >> position;
             Position pos = posFromString(position);
             board = pos;
             std::cerr << position << std::endl;
             std::cout << "game_ready" << "\n";
         } else if (current == "update") {
             //opponent made a move and we need to update the board
             std::string move_string;
             std::cin >> move_string;
             auto move = decodeMove(move_string);
             board.makeMove(move.value());
             std::cout<<"update_ready"<<"\n";
         } else if (current == "search") {
             std::cerr<<" I am searching"<<std::endl;
             std::string time_string;
             std::cin>>time_string;
             std::cerr<<"timeMove: "<<time_string<<std::endl;
             Move bestMove;
             auto value = searchValue(board, bestMove, MAX_PLY, std::stoi(time_string), false);
             auto move_string = encodeMove(bestMove);
             std::cout << move_string << "\n";
             board.makeMove(bestMove);
             std::cerr<<"I send the move"<<std::endl;
         }
     }*/
    Zobrist::initializeZobrisKeys();
    const int numEngines = 2;
    int mainPipe[numEngines][2];
    int enginePipe[numEngines][2];

    Engine engine{Engine::State::Idle, enginePipe[0][0], mainPipe[0][1]};
    engine.setTime(1000);
    Engine engine2{Engine::State::Idle, enginePipe[1][0], mainPipe[1][1]};
    engine2.setTime(1000);
    Interface inter{engine, engine2};

    std::deque<Position> openingQueue;
    std::vector<std::string> engine_paths{"old_engine", "old_engine"};


    pid_t pid;
    for (auto i = 0; i < numEngines; ++i) {
        pipe(mainPipe[i]);
        pipe(enginePipe[i]);
        pid = fork();
        if (pid < 0) {
            std::cerr << "Error" << std::endl;
            exit(EXIT_FAILURE);
        } else if (pid == 0) {
            dup2(mainPipe[i][0], STDIN_FILENO);
            dup2(enginePipe[i][1], STDOUT_FILENO);
            const std::string command = "./" + engine_paths[i];
            execlp(command.c_str(), engine_paths[i].c_str(), NULL);
            exit(EXIT_SUCCESS);
        }
    }
    if (pid > 0) {
        for (int k = 0; k < numEngines; ++k) {
            close(mainPipe[k][0]);
            close(enginePipe[k][1]);
            fcntl(enginePipe[k][0], F_SETFL, O_NONBLOCK | O_RDONLY);
        }
        inter.board = Position::getStartPosition();
        while (true) {
            inter.process();
        }


    }


    int status;
    for (int k = 0; k < numEngines; ++k) {
        wait(&status);
    }

}
