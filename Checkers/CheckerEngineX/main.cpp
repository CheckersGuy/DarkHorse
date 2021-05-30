#include <vector>
#include "Transposition.h"
#include "GameLogic.h"
#include "Perft.h"

using Game = std::pair<std::vector<Position>, int>;

Game generate_game(Position start_pos, int time_c, bool print = false) {
    std::vector<Position> history;

    TT.clear();
    Board board;
    board = start_pos;
    int result = 0;
    int i = 0;
    for (i = 0; i < 500; ++i) {
        MoveListe liste;
        getMoves(board.getPosition(), liste);
        history.emplace_back(board.getPosition());
        if (print) {
            board.printBoard();
            std::cout << std::endl;
        }

        //checking for 3 fold repetition
        auto count = std::count_if(history.begin(), history.end(), [&](Position pos) {
            return board.getPosition() == pos;
        });
        if (count >= 3) {
            result = 0;
            break;
        }

        if (liste.isEmpty()) {
            result = (board.getMover() == BLACK) ? 1 : -1;
            break;
        }

        Move best;
        searchValue(board, best, MAX_PLY, time_c, print);
        board.makeMove(best);
    }
    //What to do if we reached 500 moves ?
    //ignore all positions generated thus far
    if (i >= 499)
        history.clear();

    return std::make_pair(history, result);
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


//game-generation

#include <Network.h>

int main(int argl, const char **argc) {

    initialize();
    Board board;
    use_classical(false);


    network.load("test_open_scalxx9.weights");
    network.addLayer(Layer{120, 256});
    network.addLayer(Layer{256, 32});
    network.addLayer(Layer{32, 32});
    network.addLayer(Layer{32, 1});

    network.init();

    network2.load("test_end_scalxx9.weights");
    network2.addLayer(Layer{120, 256});
    network2.addLayer(Layer{256, 32});
    network2.addLayer(Layer{32, 32});
    network2.addLayer(Layer{32, 1});

    network2.init();






    Position pos = Position::getStartPosition();
    pos.printPosition();
    std::cout<<std::endl;


    float test = network.compute_incre_forward_pass(pos);
    network.set_input(pos);
    std::cout<<test<<std::endl;



    return 0;


   TT.resize(23);
    board = Position::getStartPosition();
    board = Position::pos_from_fen("W:W17:B4,3,K6,K14");


    board.printBoard();


    std::cout << std::endl;

    std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;

    Move best;
    searchValue(board, best, MAX_PLY, 20000000, true);
    board.makeMove(best);
    board.printBoard();
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
        } else if (current == "generate") {
            std::string fen_string;
            std::cin >> fen_string;
            std::string time_string;
            std::cin >> time_string;

            int time = std::stoi(time_string);
            Position start_pos = Position::pos_from_fen(fen_string);
            auto game = generate_game(start_pos, time, false);
            std::cout << "game_start" << "\n";
            std::cout << "result" << "\n";
            std::cout << std::to_string(game.second) << "\n";
            for (Position p : game.first) {
                if (!p.isEnd() && !p.isEmpty())
                    std::cout << p.get_fen_string() << "\n";
            }
            std::cout << "game_end" << "\n";

        }
    }


    return 0;
}