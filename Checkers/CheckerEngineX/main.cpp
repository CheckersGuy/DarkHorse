


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
#include <sys/wait.h>
#include <unistd.h>
#include <regex>

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
    return position;
}

Position posFromString(const std::string pos) {
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

    return result;
}


struct Interface {
    enum State {
        Idle, Ready, Searching
    };
    const int &engineRead1;
    const int &engineRead2;
    const int &engineWrite1;
    const int &engineWrite2;
    State oneState;
    State twoState;
    Board board;

    void initEngines();

    void writeMessage(const int &pipe, const std::string &message);

    void processInput(const int &readPipe);

    void startGame(const Position pos);
};

void Interface::initEngines() {
    std::string message = "init\n";
    writeMessage(engineWrite1, message);
    writeMessage(engineWrite2, message);
}


void Interface::startGame(const Position pos) {
    std::string message = "new_game\n";
    writeMessage(engineWrite1, message);
    writeMessage(engineWrite1, getPositionString(pos));
    writeMessage(engineWrite2, message);
    writeMessage(engineWrite2, getPositionString(pos));

}


void Interface::processInput(const int &readPipe) {
    std::string message;
    char c;
    while ((read(readPipe, &c, sizeof(char))) != -1) {
        if (c == '\n') {
            break;
        } else {
            message += c;
        }
    }

    if (message == "init_ready") {
        std::cout << "ReadyEngine" << std::endl;
    }

    if (message == "game_ready") {
        std::cout << "Engines setup position" << std::endl;
    }

    std::regex reg("[0-9]{1,2}[|][0-9]{1,2}([|][0-9]{1,2})*");

    if (std::regex_match(message, reg)) {
        //engine send a move back
        Move move;
        std::regex reg2("[^|]{2}");
        std::sregex_iterator iterator(message.begin(), message.end(), reg2);
        auto from =(*iterator++).str();
        auto to =(*iterator).str();


        move.from = 1u<<std::stoi(from);
        move.to = 1u<<std::stoi(to);

        for (auto it = iterator; it != std::sregex_iterator{}; ++it) {
            auto value =(*iterator).str();
            move.captures= 1u<<std::stoi(value);
        }
        std::cout<<"From: "<<move.getFromIndex()<<std::endl;
        std::cout<<"To: "<<move.getToIndex()<<std::endl;

    }
}

void Interface::writeMessage(const int &pipe, const std::string &message) {
    write(pipe, (char *) &message.front(), sizeof(char) * message.size());
}


int main(int argl, const char **argc) {




    Weights<double> weights;
    weights.loadWeights("/home/robin/DarkHorse/Training/cmake-build-debug/failSave.weights");
    std::cout<<"average: "<<weights.averageWeight()<<std::endl;
    std::cout<<"non-zero: "<<weights.numNonZeroValues()<<std::endl;

    Board test;
    test=Position::getStartPosition();
    test.printBoard();
    std::cout<<std::endl;

    std::cout<<"\n";

   initialize();
    setHashSize(25);
    searchValue(test,MAX_PLY,10000000,true);


/**/

/*
    std::string current;
    Board board;
    board = Position::getStartPosition();
    while (std::cin >> current) {
        if (current == "init") {
            initialize();
            setHashSize(23);
            std::cout << "init_ready" << "\n";
        } else if (current == "hashSize") {
            std::string hash;
            std::getline(std::cin, hash);
            setHashSize(std::stoi(hash));
        } else if (current == "new_game") {
            //starting a new game
            std::string position;
            std::getline(std::cin, position);
            Position pos = posFromString(position);
            board = pos;
            std::cout << "game_ready" << "\n";
            std::cerr << "Received position" << std::endl;
        } else if (current == "update") {
            //opponent made a move and we need to update the board
            std::string move;
            std::getline(std::cin, move);
        } else if (current == "search") {
            //engine is supposed to search the current position
            std::cerr << "Started search" << std::endl;
            Move bestMove;
            std::string move_string;
            auto value = searchValue(board, bestMove, MAX_PLY, 1000, false);
            move_string += std::to_string(bestMove.getFromIndex());
            move_string += "|";
            move_string += std::to_string(bestMove.getToIndex());
            if (bestMove.captures)
                move_string += "|";

            uint32_t lastMove = (bestMove.captures == 0u) ? 0u : Bits::bitscan_foward(bestMove.captures);
            uint32_t temp = bestMove.captures & (~(1u << lastMove));
            while (temp) {
                uint32_t mSquare = Bits::bitscan_foward(temp);
                temp &= temp - 1u;
                move_string += std::to_string(mSquare);
                move_string += "|";
            }
            if (lastMove) {
                move_string += std::to_string(lastMove);
            }
            std::cout << move_string << "\n";
        }
    }*/
    int numEngines = 2;
    int mainPipe[numEngines][2];
    int enginePipe[numEngines][2];
    Interface inter{enginePipe[0][0], enginePipe[1][0], mainPipe[0][1], mainPipe[1][1]};

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
            execlp("./reading", "reading", NULL);
            exit(EXIT_SUCCESS);
        }
    }
    if (pid > 0) {
        for (int k = 0; k < numEngines; ++k) {
            close(mainPipe[k][0]);
            close(enginePipe[k][1]);
        }

        inter.initEngines();
        inter.processInput(inter.engineRead1);
        inter.processInput(inter.engineRead2);
        std::string message = "search\n";
        inter.writeMessage(inter.engineWrite1, message);
        inter.writeMessage(inter.engineWrite2, message);
        inter.processInput(inter.engineRead1);
        inter.processInput(inter.engineRead2);
    }


    int status;
    for (int k = 0; k < numEngines; ++k) {
        wait(&status);
    }

}
