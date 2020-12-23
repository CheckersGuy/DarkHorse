#include <vector>
#include <bitset>
#include "Transposition.h"
#include "GameLogic.h"
#include <iterator>
#include "Perft.h"

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


    //small list of problem positions
    //1. W:WK19:B4,3,23
    //2.W:WK8:B4,K11,K10,K9,K32



    Board board;

/*


    initialize();
    Position pos = Position::pos_from_fen("B:WK4,K2:BK3,K7");
    board = Position::getStartPosition();
    board =pos;
;


    board.printBoard();

    std::cout << std::endl;



    setHashSize(27);

    std::cout<<"NonZeroWeights: "<<gameWeights.numNonZeroValues()<<std::endl;

    Move best;
    searchValue(board, best, MAX_PLY, 20000000, true);
    board.makeMove(best);
    board.printBoard();


*/



    std::string current;
    while (std::cin >> current) {
        if (current == "init") {
            TT.age_counter=0u;
            initialize();
            std::string hash_string;
            std::cin >> hash_string;
            const int hash_size = std::stoi(hash_string);
            setHashSize(hash_size);
            std::cout << "init_ready" << "\n";
        } else if (current == "new_game") {
            TT.clear();
            TT.age_counter=0u;
            board= Board{};
            std::string position;
            std::cin >> position;
            Position pos = posFromString(position);
            board = pos;
            std::cout << "game_ready" << "\n";
        } else if (current == "new_move") {
            TT.age_counter++;
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

            MoveListe liste;
            getMoves(board.getPosition(), liste);
            if (liste.length() == 1) {
                bestMove = liste[0];
            } else {
                searchValue(board, bestMove, MAX_PLY, std::stoi(time_string), false);
            }

            std::cout << "new_move" << "\n";
            std::cout << std::to_string(bestMove.getFromIndex()) << "\n";
            std::cout << std::to_string(bestMove.getToIndex()) << "\n";
            uint32_t captures = bestMove.captures;
            while (captures) {
                std::cout << std::to_string(__tzcnt_u32(captures)) << "\n";
                captures &= captures - 1u;
            }
            std::cout << "end_move" << "\n";
            board.makeMove(bestMove);
        } else if (current == "terminate") {
            //terminating the program
            break;
        }
    }

    return 0;
}