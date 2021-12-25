#include <vector>
#include "Transposition.h"
#include "GameLogic.h"
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


//game-generation

#include <Network.h>
#include <iterator>
#include <types.h>

int main(int argl, const char **argc) {
    initialize();
    Board board;
    use_classical(false);

    Statistics::mPicker.init();


/*
    network.load("modeltest2.weights");
    network.addLayer(Layer{120, 512});
    network.addLayer(Layer{512, 16});
    network.addLayer(Layer{16, 32});
    network.addLayer(Layer{32, 1});

    network.init();
*/



    network.load("testx1.weights");
    network.addLayer(Layer{120, 256});
    network.addLayer(Layer{256, 32});
    network.addLayer(Layer{32, 32});
    network.addLayer(Layer{32, 1});
    network.init();

    network2.load("testx1.weights");
    network2.addLayer(Layer{120, 256});
    network2.addLayer(Layer{256, 32});
    network2.addLayer(Layer{32, 32});
    network2.addLayer(Layer{32, 1});

    network2.init();
/*

    network.load("verybig.weights");
    network.addLayer(Layer{120, 1024});
    network.addLayer(Layer{1024, 16});
    network.addLayer(Layer{16, 32});
    network.addLayer(Layer{32, 1});

    network.init();
*/







    TT.resize(23);
    board = Position::getStartPosition();
    //board = Position::pos_from_fen("W:W9,29:BK3,K6,K12");
    board = Position::pos_from_fen("W:WK6:B4,3");
    board.printBoard();

    Move best;
    searchValue(board, best, MAX_PLY, 100000000, true);
    board.makeMove(best);
    board.printBoard();
    MoveListe liste;
    getMoves(board.getPosition(), liste);







    std::string current;
    while (std::cin >> current) {
        if (current == "init") {
            TT.age_counter = 0u;
            initialize();
            std::string hash_string;
            std::cin >> hash_string;
            const int hash_size = std::stoi(hash_string);
            TT.resize(hash_size);
            std::cout << "init_ready" << "\n";
        } else if (current == "new_game") {
            TT.clear();
            TT.age_counter = 0u;
            board = Board{};
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
            searchValue(board, bestMove, MAX_PLY, std::stoi(time_string), false);
            std::cout << "new_move" << "\n";
            std::cout << std::to_string(bestMove.getFromIndex()) << "\n";
            std::cout << std::to_string(bestMove.getToIndex()) << "\n";
            uint32_t captures = bestMove.captures;
            while (captures) {
                std::cout << std::to_string(Bits::bitscan_foward(captures)) << "\n";
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