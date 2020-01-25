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

int main(int argl, const char **argc) {
    std::string current;
    Position start_pos = Position::getStartPosition();

    Board board;
 /*   board = Position::getStartPosition();
    board.printBoard();
    std::cout<<std::endl;

    initialize();
    setHashSize(25);
    Move best;
    searchValue(board,best, 128, 300000, true);
    board.makeMove(best);
    board.printBoard();

    while (true) {
        MoveListe liste;
        getMoves(board.getPosition(), liste);
        if (board.isRepetition() || liste.length() == 0)
            break;

        Move best;
        auto value = searchValue(board, best, MAX_PLY, 300, false);
        board.makeMove(best);
        board.printBoard();
        std::cout << "Value: " << value << "\n" << std::endl;
    }
*/

    while (std::cin >> current) {
        if (current == "init") {
            initialize();
            std::string hash_string;
            std::cin >> hash_string;
            const int hash_size = std::stoi(hash_string);
            setHashSize(hash_size);
            std::cerr << "HashSize: " << hash_string << std::endl;
            std::cout << "init_ready" << "\n";
        } else if (current == "new_game") {
            std::string position;
            std::cin >> position;
            Position pos = posFromString(position);
            board = pos;
            std::cout << "game_ready" << "\n";
        } else if (current == "new_move") {
            //opponent made a move and we need to update the board
            Move move;
            std::vector<uint32_t> squares;
            std::string line;
            std::cin >> line;
            while (!line.empty()) {
                if (line == "end_move")
                    break;
                squares.emplace_back(std::stoi(line));
                std::cerr << "Line " << line << std::endl;
                std::cin >> line;
            }
            move.from = 1u << squares[0];
            move.to = 1u << squares[1];
            for (auto i = 2; i < squares.size(); ++i) {
                move.captures |= 1u << squares[i];
            }
            board.makeMove(move);
            std::cout << "update_ready" << "\n";
        } else if (current == "search") {
            std::string time_string;
            std::cin >> time_string;
            Move bestMove;
            auto value = searchValue(board, bestMove, MAX_PLY, std::stoi(time_string), false);
            std::cerr << "Value: " << value << std::endl;
            std::cout << "new_move" << "\n";
            std::cout << std::to_string(__tzcnt_u32(bestMove.from)) << "\n";
            std::cout << std::to_string(__tzcnt_u32(bestMove.to)) << "\n";
            uint32_t captures = bestMove.captures;
            while (captures) {
                std::cout << std::to_string(__tzcnt_u32(captures)) << "\n";
                captures &= captures - 1u;
            }
            std::cout << "end_move" << "\n";
            board.makeMove(bestMove);
        }
    }


}