#include <vector>
#include <future>
#include <bitset>
#include <map>
#include "Transposition.h"
#include "GameLogic.h"
#include "BoardFactory.h"
#include "Perft.h"
#include <list>
#include <iterator>
#include <regex>

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


int main() {

        std::string current;
       Board board;
       while (std::cin >> current) {
           if (current == "init") {
               initialize();
               std::string hash_string;
               std::cin >> hash_string;
               const int hash_size = std::stoi(hash_string);
               setHashSize(1u << hash_size);
               std::cerr << "HashSize: " << hash_string << std::endl;
               std::cout << "init_ready" << "\n";
           } else if (current == "new_game") {
               std::string position;
               std::cin >> position;
               Position pos = posFromString(position);
               BoardFactory::setUpPosition(board,pos);
               std::cerr << position << std::endl;
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
                   std::cin >> line;
               }
               move.setFrom(squares[0]);
               move.setTo(squares[1]);
               for (auto i = 2; i < squares.size(); ++i) {
                   move.captures |= 1u << squares[i];
               }
               uint32_t sq =1u<<move.getTo();
               if((sq & board.getPosition()->K)!=0u){
                   move.setPieceType(1u);
               }
               board.makeMove(move);
               std::cout << "update_ready" << "\n";
           } else if (current == "search") {
               std::string time_string;
               std::cin >> time_string;
               Move bestMove;
               auto value = searchValue(board, bestMove, MAX_PLY, std::stoi(time_string), false);
               std::cout << "new_move" << "\n";
               std::cout << std::to_string(bestMove.getFrom()) << "\n";
               std::cout << std::to_string(bestMove.getTo()) << "\n";
               uint32_t captures = bestMove.captures;
               while (captures) {
                   std::cout << std::to_string(__tzcnt_u32(captures))<<"\n";
                   captures &= captures - 1u;
               }
               std::cout << "end_move" << "\n";
               board.makeMove(bestMove);
           }
       }
}