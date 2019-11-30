


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
        uint32_t current = 1u << S[i];
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
        uint32_t current = 1u << S[i];
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
};

void Interface::initEngines() {
    std::string message = "init\n";
    writeMessage(engineWrite1, message);
    writeMessage(engineWrite2, message);
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
    std::regex reg("[0-9]{1,2}[|][0-9]{1,2}([|][0-9]{1,2})*");

    if (message == "ready") {
        std::cout << "ReadyEngine" << std::endl;
    }
    if (std::regex_match(message, reg)) {
        //engine send a move back
        Move move;
        std::regex reg2("[0-9]{1,2}[^|]");
        std::sregex_iterator iterator(message.begin(), message.end(), reg2);
        move.from = 1u << (std::stoi((*iterator++).str()));
        move.to = 1u << (std::stoi((*iterator++).str()));
        for (auto it = iterator; it != std::sregex_iterator{}; ++it) {
            move.captures |= 1u << (std::stoi((*it).str()));
        }
        //Later we need to check if its a legal move
        board.makeMove(move);
    }
}

void Interface::writeMessage(const int &pipe, const std::string &message) {
    write(pipe, (char *) &message.front(), sizeof(char) * message.size());
}

#include "Weights.h"
#include <immintrin.h>
#include <x86intrin.h>

int main(int argl, const char **argc) {

    /*std::array<int,4>values{1,1,1,1};
    compress<uint8_t >(values.begin(),values.end(),"test.compressed");

*/

   /* std::array<int,4>values;
    decompress<uint8_t,int>("test.compressed",values.begin());
*/


/*

    initialize();
    setHashSize(23);
    Board test;
    test = Position::getStartPosition();

    searchValue(test, MAX_PLY, 700000, true);


*/
    std::cout<<"Perft 12 timeTest"<<std::endl;
    auto time=0;
    for(int i=0;i<10;++i){
        Board board;
        board=Position::getStartPosition();
        auto t1=std::chrono::high_resolution_clock::now();
        Perft::perftCheck(board,14);
        auto t2=std::chrono::high_resolution_clock::now();
        auto c=t2-t1;
        time+=c.count();
        std::cout<<"Time: "<<c.count()/1000000<<std::endl;
    }
   

   /* std::string current;
    Board board;
    while (std::cin >> current) {
        if (current == "init") {
            initialize();
            setHashSize(23);
            std::cout << "ready" << "\n";
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
        } else if (current == "update") {
            //opponent made a move and we need to update the board
            std::string move;
            std::getline(std::cin, move);
        } else if (current == "search") {
            //engine is supposed to search the current position
            Move bestMove;
            std::string move_string;
            auto value = searchValue(board, bestMove, MAX_PLY, 1000, false);
            move_string += std::to_string(bestMove.getFrom());
            move_string += "|";
            move_string += std::to_string(bestMove.getTo());
            move_string += "|";
            uint32_t lastMove = Bits::bitscan_foward(bestMove.captures);
            uint32_t temp = bestMove.captures & (~(1u << lastMove));
            while (temp) {
                uint32_t mSquare = Bits::bitscan_foward(temp);
                temp &= temp - 1u;
                move_string += std::to_string(mSquare);
                move_string += "|";
            }
            move_string += std::to_string(lastMove);
            std::cout << move_string << "\n";
        }
    }
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
        int numGames = 0;
        std::string openingPath = "";
        std::vector<Position> openings;


        for (int k = 0; k < numEngines; ++k) {
            close(mainPipe[k][0]);
            close(enginePipe[k][1]);
        }
        std::string message = "new_game\n";
        inter.initEngines();
        std::string posString = getPositionString(inter.board.getPosition());
        inter.writeMessage(inter.engineWrite1, message);
        inter.writeMessage(inter.engineWrite1, posString);
        inter.writeMessage(inter.engineWrite2, message);
        inter.writeMessage(inter.engineWrite2, posString);
    }


    int status;
    for (int k = 0; k < numEngines; ++k) {
        wait(&status);
    }*/

}
