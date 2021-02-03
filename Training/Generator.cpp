//
// Created by root on 03.02.21.
//

#include "Generator.h"

void Generator::start() {
    //creating pipes
    constexpr int max_parallelism = 128;
    int read_pipes[max_parallelism][2];
    int write_pipes[max_parallelism][2];

    pid_t pid;
    for (auto i = 0; i < parallelism; ++i) {
        pipe(read_pipes[i]);
        pipe(write_pipes[i]);
        pid = fork();
        if (pid < 0) {
            std::cerr << "Error" << std::endl;
            std::exit(-1);
        }
        if (pid == 0) {
            //child process
            close(read_pipes[i][0]);
            close(write_pipes[i][1]);
            std::cout << "I am a child process" << std::endl;

            std::pair<std::vector<Position>,int> game = generate_game(Position::getStartPosition(),50);
            for(auto p : game){

            }

            return;
        }
    }

    if (pid > 0) {
        //main_process
        for (auto i = 0; i < parallelism; ++i) {
            close(read_pipes[i][1]);
            close(write_pipes[i][0]);
        }

        for(auto counter =0;counter<num_games;++counter){

        }

    }


}

std::pair<std::vector<Position>, int> Generator::generate_game(Position start_pos, int time_c) {
    std::vector<Generator::Sample> samples;
    std::vector<Position> history;
    Board board;
    board = start_pos;
    int result = 0;
    for (auto i = 0; i < 400; ++i) {
        //checking if we have a winning position
        MoveListe liste;
        getMoves(board.getPosition(), liste);
        if (liste.isEmpty()) {
            //winning position
            if (board.getMover() == BLACK)
                result = 1;
            else
                result = -1;
            break;
        }
        //checking for 3 fold repetition
        Position current = board.getPosition();

        history.emplace_back(board.getPosition());
        Move best;
        searchValue(board, best, MAX_PLY, 50, false);
        board.makeMove(best);
    }
    return std::make_pair(history, result);
}
