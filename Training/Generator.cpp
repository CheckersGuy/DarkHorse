//
// Created by root on 03.02.21.
//

#include "Generator.h"
#include "Utilities.h"

std::optional<std::vector<Position>> generate_game(Position start_pos, int time_c, int &result) {
    TT.clear();
    std::vector<Position> history;
    result = 0;
    Board board;
    board = start_pos;
    int i;
    for (i = 0; i < 400; ++i) {
        history.emplace_back(board.getPosition());
        MoveListe liste;
        getMoves(board.getPosition(), liste);
        if (liste.isEmpty()) {
            if (board.getPosition().getColor() == BLACK) {
                result = 1;
            } else {
                result = -1;
            }
            break;
        }

        if (liste.length() == 1) {
            board.makeMove(liste[0]);
            continue;
        }

        //checking for 3 fold repetition
        const auto current = board.getPosition();
        auto count = std::count_if(history.begin(), history.end(), [&current](Position pos) {
            return current == pos;
        });
        if (count >= 3)
            break;

        Move best;
        searchValue(board, best, MAX_PLY, time_c, false);
        board.makeMove(best);
    }
    if (i < 400)
        return std::make_optional(history);
    return std::nullopt;
}

void Generator::clearBuffer() {
    if (buffer.size() < BUFFER_SIZE)
        return;
    std::cout << "Cleared Buffer" << std::endl;
    std::ofstream stream(output, std::ios::binary | std::ios::app);
    //same format that's being used in the pytorch trainer
    auto lambda = [&](Sample s) {
        Position pos;
        pos = s.first;
        stream.write((char *) &pos.WP, sizeof(uint32_t));
        stream.write((char *) &pos.BP, sizeof(uint32_t));
        stream.write((char *) &pos.K, sizeof(uint32_t));

        int color = (pos.color == BLACK) ? 0 : 1;
        int result = s.second;

        stream.write((char *) &color, sizeof(int));
        stream.write((char *) &result, sizeof(int));
    };
    std::for_each(buffer.begin(), buffer.end(), lambda);
    buffer.clear();
    stream.close();
}

void Generator::start() {

    //opening book

    std::vector<Position> openings;
    std::ifstream stream("../Training/Positions/new.book");
    std::istream_iterator<Position> begin(stream);
    std::istream_iterator<Position> end;
    std::copy(begin, end, std::back_inserter(openings));

    //creating pipes
    constexpr int max_parallelism = 128;
    int read_pipes[max_parallelism][2];
    int write_pipes[max_parallelism][2];
    std::array<bool, max_parallelism> is_busy{false};

    pid_t pid;
    for (auto i = 0; i < parallelism; ++i) {
        pipe(read_pipes[i]);
        pipe(write_pipes[i]);
        pid = fork();
        if (pid < 0) {
            std::cerr << "Error" << std::endl;
            std::exit(-1);
        } else if (pid == 0) {
            //child process
            initialize();
            setHashSize(21);
            while (true) {
                Position pos;
                read(write_pipes[i][0], (char *) &pos, sizeof(Position));
                if (pos.isEmpty())
                    std::exit(1);
                int result;
                auto game = generate_game(pos, 20, result);
                if (!game.has_value()) {
                    Sample s{};
                    write(read_pipes[i][1], (char *) &s, sizeof(Sample));
                } else {
                    for (Position p : game.value()) {
                        Sample s = std::make_pair(p, result);
                        write(read_pipes[i][1], (char *) &s, sizeof(Sample));
                    }
                }

            }
            std::exit(1);

        }
    }

    if (pid > 0) {
        //main_process
        for (auto i = 0; i < parallelism; ++i) {
            close(read_pipes[i][1]);
            close(write_pipes[i][0]);
            fcntl64(read_pipes[i][0], F_SETFL, O_NONBLOCK | O_RDONLY);
        }

        int counter = 0;
        int index = 0;
        while (counter < num_games) {
            for (auto k = 0; k < parallelism; ++k) {
                if (!is_busy[k]) {
                    Position start_pos;
                    if (index >= openings.size() - 1)
                        index = 0;
                    start_pos = openings[index];
                    start_pos.printPosition();
                    write(write_pipes[k][1], (char *) &start_pos, sizeof(Position));
                    is_busy[k] = true;
                    index++;
                } else {
                    //trying to read samples
                    Sample s;
                    int bytes_read;
                    do {
                        bytes_read = read(read_pipes[k][0], (char *) &s, sizeof(Sample));
                        if (!s.first.isEmpty()) {
                            if (is_busy[k]) {
                                std::cout << ++counter << std::endl;
                            }

                            if (!s.first.isEnd()) {
                                buffer.emplace_back(s);
                            }
                            is_busy[k] = false;
                        }
                    } while (bytes_read != -1);
                }
            }
            clearBuffer();
        }

        for (int p = 0; p < parallelism; ++p) {
            Position pos{};
            write(write_pipes[p][1], (char *) &pos, sizeof(Position));
        }

        //at the end we need for the child-processes
        int status;
        for (int p = 0; p < parallelism; ++p) {
            wait(&status);
        }
    }


}

void Generator::set_num_games(size_t num) {
    num_games = num;
}

void Generator::set_parallelism(size_t threads) {
    parallelism = threads;
}

