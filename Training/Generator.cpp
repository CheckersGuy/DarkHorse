//
// Created by root on 03.02.21.
//

#include "Generator.h"

bool Sample::operator==(const Sample &other) const {
    if (position != other.position) {
        return false;
    }
    return result == other.result;
}

bool Sample::operator!=(const Sample &other) const {
    return !((*this) == other);
}

std::ostream &operator<<(std::ostream &stream, const Sample &s) {
    stream << s.position;
    stream.write((char *) &s.result, sizeof(int));
    return stream;
}

std::istream &operator>>(std::istream &stream, Sample &s) {
    stream >> s.position;
    int result;
    stream.read((char *) &result, sizeof(int));
    s.result = result;
    return stream;
}

void Instance::write_message(std::string msg) {
    const std::string ex_message = msg + "\n";
    write(write_pipe, (char *) &ex_message.front(), sizeof(char) * (ex_message.size()));
}

void Generator::set_time(int time) {
    time_control = time;
}

void Generator::print_stats() {
    std::cout << "num_games: " << game_counter << "\n";
    std::cout << "num_wins: " << num_wins << "\n";
    double ratio = ((double) num_wins) / ((double) game_counter);
    std::cout << "decisive_ratio: " << ratio << "\n";
    std::cout << "\n";
    std::cout << "\n";
    std::flush(std::cout);
}

void Generator::clearBuffer() {
    std::string path{"../Training/TrainData/"};
    path.append(output);
    std::ofstream stream(path, std::ios::binary | std::ios::app);
    if (!stream.good()) {
        std::cerr << "Could not clear buffer" << std::endl;
        std::exit(-1);
    }
    std::copy(buffer.begin(), buffer.end(), std::ostream_iterator<Sample>(stream));
    //appending to the file
    std::cout << "Cleared buffer" << std::endl;
    buffer.clear();
}

Position Generator::get_start_pos() {
    if (opening_index >= openings.size() - 1) {
        opening_index = 0;
    }
    return openings[opening_index++];
}

std::string Instance::read_message() {
    std::string message;
    char c;
    int result;
    do {
        result = read(read_pipe, (char *) (&c), sizeof(char));
        if (c == char(0) || c == '\n') {
            break;
        } else {
            message += c;
        }

    } while (result != -1);
    return message;
}

void Generator::set_hash_size(int size) {
    hash_size = size;
}

void Generator::process() {
    for (auto &instance : instances) {
        if (instance.state == Instance::Idle) {
            if (!instance.waiting_response) {
                instance.write_message("init");
                instance.write_message(std::to_string(hash_size));
                instance.waiting_response = true;
            }
            auto msg = instance.read_message();
            if (msg == "init_ready") {
                std::cout << "initialized engine" << std::endl;
                instance.state = Instance::Init;
                instance.waiting_response = false;
            }
        }
        if (instance.state == Instance::Init) {
            if (!instance.waiting_response) {
                Position start_pos = get_start_pos();
                instance.write_message("generate");
                instance.write_message(start_pos.get_fen_string());
                instance.write_message(std::to_string(time_control));
                instance.waiting_response = true;
            }
            auto msg = instance.read_message();
            if (msg == "game_start") {
                int result = 1000;
                std::cout << "Receiving game" << std::endl;
                do {
                    msg = instance.read_message();
                    if (msg != "game_end") {
                        Position pos;
                        if (msg == "result") {
                            msg = instance.read_message();
                            result = std::stoi(msg);
                        } else {
                            if (!msg.empty()) {
                                try {
                                    pos = Position::pos_from_fen(msg);
                                    Sample s;
                                    s.position = pos;
                                    s.result = result;
                                    if (result != 1000)
                                        buffer.emplace_back(s);
                                    /*     pos.printPosition();
                                     std::cout << result << std::endl;
                                     std::cout << std::endl;*/
                                } catch (std::domain_error &error) {
                                    //write something to a log file
                                    //I don't expect this to happen very often though
                                    std::ofstream stream("Generator_log", std::ios::app);
                                    stream << "Error occured when parsing the message '" << msg << "'";
                                    stream << "\n" << "\n";
                                    stream.close();
                                }

                            }
                        }
                    }
                } while (msg != "game_end");
                if (result == -1 || result == 1)
                    num_wins++;
                if (buffer.size() >= BUFFER_SIZE) {
                    clearBuffer();
                }
                game_counter++;
                print_stats();
                instance.waiting_response = false;
            }
        }

    }

}

void Generator::start() {

    //opening book
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
        } else if (pid == 0) {
            //child process
            close(read_pipes[i][0]);
            close(write_pipes[i][1]);
            dup2(write_pipes[i][0], STDIN_FILENO);
            dup2(read_pipes[i][1], STDOUT_FILENO);
            //will be changed to call a different executable
            //will work on that tomorrow
            std::string e = "../Training/Engines/";
            e += engine_path;
            const std::string command = "./" + e;
            auto result = execlp(command.c_str(), e.c_str(), NULL);
            std::exit(result);
        }
    }

    if (pid > 0) {

        for (auto i = 0; i < parallelism; ++i) {
            instances.emplace_back(Instance{read_pipes[i][0], write_pipes[i][1]});
        }

        //main_process
        for (auto i = 0; i < parallelism; ++i) {
            close(read_pipes[i][1]);
            close(write_pipes[i][0]);
            fcntl64(read_pipes[i][0], F_SETFL, O_NONBLOCK | O_RDONLY);
        }
        while (game_counter < num_games) {
            process();
        }

        for (auto &instance : instances) {
            instance.write_message("terminate");
        }
        clearBuffer();


        //at the end we need for the child-processes
        int status;
        for (int p = 0; p < parallelism; ++p) {
            wait(&status);
        }
    }


}

void Generator::startx() {
    //Positions to be saved to a file
    const size_t num_positions = 10000000;

    std::cout << "Number of openings: " << openings.size() << std::endl;
    int *counter;
    counter = (int *) mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
                           0);

    *counter = 0;

    pthread_mutex_t *pmutex = NULL;
    pthread_mutexattr_t attrmutex;
/* Allocate memory to pmutex here. */
    pmutex = (pthread_mutex_t *) mmap(NULL, sizeof(pthread_mutex_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,
                                      -1, 0);

/* Initialise mutex. */
    pthread_mutex_init(pmutex, &attrmutex);

    pid_t id;
    for (auto i = 0; i < parallelism; ++i) {
        id = fork();
        if (id < 0) {
            std::cerr << "Could not fork the main process" << std::endl;
            std::exit(-1);
        }
        if (id == 0) {
            std::vector<Sample> local_buffer;
            //child takes a position and generates games
            const uint64_t seed = 13199312313ull + 23ull * i;
            initialize(seed);
            use_classical(true);
            TT.resize(hash_size);
            //play a game and increment the opening-counter once more

            int opening_counter = 0;

            int move_count;
            while (true) {
                pthread_mutex_lock(pmutex);
                if (*counter >= num_positions) {
                    break;
                }
                Position opening = openings[opening_counter];
                Board board;
                board = opening;
                /* std::cout << "OpeningCounter: " << *opening_counter << std::endl;
                 board.printBoard();
                 std::cout << std::endl;
 */
                pthread_mutex_unlock(pmutex);
                std::vector<Sample> game;
                for (move_count = 0; move_count < 600; ++move_count) {
                    MoveListe liste;
                    getMoves(board.getPosition(), liste);

                    if (liste.length() == 0) {
                        //end of the game, a player won
                        for (auto &sample : game) {
                            sample.result = (board.getMover() == BLACK) ? 1 : -1;
                        }
                        break;
                    }
                    if (board.isRepetition()) {
                        //draw by repetition
                        for (auto &sample : game) {
                            sample.result = 0;
                        }
                        break;
                    }

                    Move best;
                    if (liste.length() == 1) {
                        best = liste[0];
                    } else {
                        searchValue(board, best, MAX_PLY, time_control, false);
                    }
                    board.makeMove(best);

                    Sample current;
                    current.position = board.getPosition();
                    game.emplace_back(current);
                }
                pthread_mutex_lock(pmutex);
                std::copy(game.begin(), game.end(), std::back_inserter(local_buffer));
                if (local_buffer.size() >= 2000) {
                    std::cout << "ClearedBuffer" << std::endl;
                    Utilities::write_to_binary<Sample>(local_buffer.begin(), local_buffer.end(), output,
                                                       std::ios::app | std::ios::binary);
                    local_buffer.clear();
                }
                *counter = *counter + (int) game.size();
                std::cout << "Counter: " << *counter << std::endl;
                pthread_mutex_unlock(pmutex);
                opening_counter++;
            }
            std::exit(1);
        }
    }

    if (id > 0) {
        //main_process
        wait(NULL);
    }




    /* Clean up. */
    pthread_mutex_destroy(pmutex);
    pthread_mutexattr_destroy(&attrmutex);
}


void Generator::set_num_games(size_t num) {
    num_games = num;
}

void Generator::set_parallelism(size_t threads) {
    parallelism = threads;
}

