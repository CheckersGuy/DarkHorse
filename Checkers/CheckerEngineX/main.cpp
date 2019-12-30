


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


std::optional<Move> decodeMove(const std::string &move_string) {
    std::regex reg("[0-9]{1,2}[|][0-9]{1,2}([|][0-9]{1,2})*");
    if (move_string.size() > 2)
        if (std::regex_match(move_string, reg)) {
            Move result;
            std::regex reg2("[^|]+");
            std::sregex_iterator iterator(move_string.begin(), move_string.end(), reg2);
            auto from = (*iterator++).str();
            auto to = (*iterator++).str();
            result.setFrom(std::stoi(from));
            result.setTo(std::stoi(to));
            for (auto it = iterator; it != std::sregex_iterator{}; ++it) {
                auto value = (*it).str();
                result.captures |= 1u << std::stoi(value);
            }
            return std::make_optional(result);
        }
    return std::nullopt;
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

std::string encodeMove(Move move) {
    std::string move_string;
    move_string += std::to_string(move.getFrom());
    move_string += "|";
    move_string += std::to_string(move.getTo());
    if (move.captures)
        move_string += "|";

    uint32_t lastMove = (move.captures ==0u)?0u : _tzcnt_u32(move.captures);
    uint32_t temp = move.captures & (~(1u << lastMove));
    while (temp) {
        uint32_t mSquare = _tzcnt_u32(temp);
        temp &= temp - 1u;
        move_string += std::to_string(mSquare);
        move_string += "|";
    }
    if (lastMove) {
        move_string += std::to_string(lastMove);
    }
    return move_string;
}


int main() {

    std::string current;
    Board board;
    BoardFactory::setUpStartingPosition(board);
    while (std::cin >> current) {
        if (current == "init") {
            initialize();
            setHashSize(21);
            std::string hash_string;
            std::cin >> hash_string;
            const int hash_size = std::stoi(hash_string);
            setHashSize(1u << hash_size);
            std::cerr << "HashSize: " << hash_string << std::endl;
            std::cout << "init_ready" << "\n";
        } else if (current == "new_game") {
            std::cerr<<"new_game"<<std::endl;
            std::string position;
            std::cin >> position;
            Position pos =posFromString(position);
            BoardFactory::setUpPosition(board,pos);
            std::cerr << position << std::endl;
            std::cout << "game_ready" << "\n";
        } else if (current == "update") {
            //opponent made a move and we need to update the board
            std::string move_string;
            std::cin >> move_string;
            auto move = decodeMove(move_string);
            board.makeMove(move.value());
            std::cout << "update_ready" << "\n";
        } else if (current == "search") {
            std::cerr << " old_engine searching" << std::endl;
            std::string time_string;
            std::cin >> time_string;
            std::cerr << "timeMove: " << time_string << std::endl;
            Move bestMove;
            auto value = searchValue(board, bestMove, MAX_PLY, std::stoi(time_string), false);
            auto move_string = encodeMove(bestMove);
            std::cout << move_string << "\n";
            std::cerr << "I send the move" << std::endl;
            std::cerr<<move_string<<std::endl;
        }
    }
}