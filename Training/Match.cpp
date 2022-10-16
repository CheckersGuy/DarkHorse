//
// Created by robin on 7/26/18.
//

#include "Match.h"


bool Interface::is_n_fold(int n) {
    if (history.empty())
        return false;

    const Position other = history.back();
    int count = 0;
    for (int i = (int) history.size() - 2; i >= 0; --i) {
        if (history[i] == other)
            count++;
        if (count >= n)
            return true;
    }

    return false;
}


bool Interface::is_legal_move(Move move) {
    MoveListe liste;
    get_moves(pos, liste);
    Position check_pos = pos;
    check_pos.make_move(move);
    for (auto m : liste.liste) {
        Position copy = pos;
        copy.make_move(m);
        if (copy == check_pos)
            return true;
    }
    return false;
}


void Engine::setTime(int time) {
    time_move = time;
}

void Engine::setHashSize(int hash) {
    hash_size = hash;
}

std::string Engine::readPipe() {
    std::string message;
    char c;
    int result;
    do {
        result = read(engineRead, (char *) (&c), sizeof(char));
        if (c == char(0) || c == '\n') {
            break;
        } else {
            message += c;
        }

    } while (result != -1);
    return message;
}


void Engine::writeMessage(const std::string &msg) {
    const std::string ex_message = msg + "\n";
    write(engineWrite, (char *) &ex_message.front(), sizeof(char) * ex_message.size());
}

Move Engine::search() {
    if (state == State::Game_Ready) {
        if (!waiting_response) {
            writeMessage("search");
            writeMessage(std::to_string(time_move));
            waiting_response = true;
        }

        auto answer = readPipe();
        if (answer == "new_move") {
            Move move;
            auto line = readPipe();
            std::vector<uint32_t> squares;
            while (!line.empty()) {
                if (line == "end_move")
                    break;
                squares.emplace_back(std::stoi(line));
                line = readPipe();
            }
            waiting_response = false;
            move.from = 1u << squares[0];
            move.to = 1u << squares[1];
            for (auto i = 2; i < squares.size(); ++i) {
                move.captures |= 1u << squares[i];
            }

            return move;
        }
    }
    return Move{};
}


void Engine::newGame(const Position &pos) {
    if (state == State::Init_Ready) {
        if (!waiting_response) {
            writeMessage("new_game");
            writeMessage(getPositionString(pos));
        }
        waiting_response = true;
        auto answer = readPipe();
        if (answer == "game_ready") {
            state = State::Game_Ready;
            waiting_response = false;
        }
    }
}


void Engine::update() {
    if (state == State::Update) {
        auto answer = readPipe();
        if (answer == "update_ready") {
            waiting_response = false;
            state = State::Game_Ready;
        }
    }
}

void Engine::initEngine() {
    if (state == State::Idle) {
        if (!waiting_response) {
            writeMessage("init");
            writeMessage(std::to_string(hash_size));
        }
        waiting_response = true;
        auto answer = readPipe();
        if (answer == "init_ready") {
            state = State::Init_Ready;
            waiting_response = false;
        }
    }

}

bool Interface::is_terminal_state() {
    if (is_n_fold(3) || history.size() >= 400)
        return true;

    MoveListe liste;
    get_moves(pos, liste);
    return liste.length() == 0;

}

void Engine::new_move(Move move) {
    writeMessage("new_move");
    writeMessage(std::to_string(move.get_from_index()));
    writeMessage(std::to_string(move.get_to_index()));
    uint32_t captures = move.captures;
    while (captures) {
        writeMessage(std::to_string(__tzcnt_u32(captures)));
        captures &= captures - 1u;
    }
    writeMessage("end_move");
}


void Interface::process() {
    for (auto &engine : engines) {
        engine.initEngine();
        engine.newGame(pos);
        engine.update();
    }
    const int second_mover = (first_mover == 0) ? 1 : 0;

    //checking if there is only one possible move
    MoveListe liste;
    get_moves(pos, liste);

    Move move{};

    move = engines[first_mover].search();


    if (!move.is_empty()) {

        if (!Interface::is_legal_move(move)) {
            std::exit(EXIT_FAILURE);
        }
        pos.make_move(move);
        history.emplace_back(pos);
        engines[second_mover].new_move(move);
        first_mover = second_mover;
    }
}


int Match::getMaxGames() {
    return maxGames;
}

void Match::setMaxGames(int games) {
    this->maxGames = games;
}


void Match::setTime(int time) {
    this->time = time;
}

void Match::setOpeningBook(std::string book) {
    this->openingBook = book;
}

const std::string &Match::getOpeningBook() {
    return openingBook;
}

void Match::setNumThreads(int threads) {
    this->threads = threads;
}

int Match::getNumThreads() {
    return threads;
}

void Match::setHashSize(int hash) {
    this->hash_size = hash;
}

void Interface::reset_engines() {
    for (auto &engine : engines) {
        engine.state = Engine::State::Idle;
    }
    pos = Position{};
}


void Interface::terminate_engines() {
    for (auto &engine : engines) {
        engine.writeMessage("terminate");
    }
}

Position Match::get_start_pos() {
    if (opening_counter >= positions.size() - 1)
        opening_counter = 0u;

    return positions[opening_counter++];
}


void Match::start() {
    //this probably needs to be reworked
    system("echo ' \\e[1;31m Engine Match \\e[0m' ");
    std::cout << "Engine1: " << first << std::endl;
    std::cout << "Engine2: " << second << std::endl;

    Zobrist::init_zobrist_keys();
    const int numEngines = 2;
    const int num_matches = this->threads;

    std::vector<Interface> interfaces;
    std::cout << "OpeningBook: " << openingBook << std::endl;
    std::cout << "Number of Positions: " << positions.size() << std::endl;
    std::vector<std::string> engine_paths{first, second};


    int mainPipe[num_matches][numEngines][2];
    int enginePipe[num_matches][numEngines][2];
    int game_count = -num_matches;

    for (int p = 0; p < num_matches; ++p) {
        Engine engine{first, Engine::State::Idle, enginePipe[p][0][0], mainPipe[p][0][1]};
        engine.setTime(times.first);
        engine.setHashSize(hash_size);
        Engine engine2{second, Engine::State::Idle, enginePipe[p][1][0], mainPipe[p][1][1]};
        engine2.setTime(times.second);
        engine2.setHashSize(hash_size);
        interfaces.emplace_back(Interface{engine, engine2});
    }
    auto& first = interfaces.front();
    std::cout<<"Time: "<<times.first<<" "<<times.second<<std::endl;

    pid_t pid;
    for (int p = 0; p < num_matches; ++p) {
        for (auto i = 0; i < numEngines; ++i) {
            pipe(mainPipe[p][i]);
            pipe(enginePipe[p][i]);
            pid = fork();
            if (pid < 0) {
                std::cerr << "Error" << std::endl;
                exit(EXIT_FAILURE);
            } else if (pid == 0) {
                dup2(mainPipe[p][i][0], STDIN_FILENO);
                dup2(enginePipe[p][i][1], STDOUT_FILENO);
                const std::string e = "../Training/Engines/" + engine_paths[i];
                const std::string command = "./" + e;
                auto argument = (i==0)?arg1 : arg2;
                if(argument.empty()) {

                    execlp(command.c_str(), e.c_str(),NULL);
                } else {

                    execlp(command.c_str(), e.c_str(),argument.c_str());
                }

                exit(EXIT_SUCCESS);
            }
        }
    }
    if (pid > 0) {
        for (int p = 0; p < num_matches; ++p) {
            for (int k = 0; k < numEngines; ++k) {
                close(mainPipe[p][k][0]);
                close(enginePipe[p][k][1]);
                fcntl64(enginePipe[p][k][0], F_SETFL, O_NONBLOCK | O_RDONLY);
            }
        }
        printf("\r");
        printf("\r");
        printf("%-5s %-5s %-5s %-5s \n", "Wins_one", "Wins_two", "Draws");
        printf("%-5d %-5d %-5d %-5d", wins_one, wins_two, draws);

        while (game_count < maxGames) {
            for (auto &inter : interfaces) {
                if (inter.pos.is_empty()) {
                    if (inter.first_game) {
                        inter.first_mover = 0;
                        inter.start_pos = get_start_pos();
                        inter.first_game = false;
                    } else {
                        inter.first_mover = 1;
                        inter.first_game = true;
                    }
                    inter.pos = inter.start_pos;


                }

                if (inter.is_terminal_state()) {
                    game_count++;
                    printf("\r");
                    printf("\r");
                    printf("%-5d %-5d %-5d", wins_one, wins_two, draws);
                    std::cout << std::endl;
                    MoveListe liste;
                    get_moves(inter.pos, liste);
                    if (liste.length() == 0) {
                        if (inter.first_mover == 0) {
                            wins_two++;
                        } else {
                            wins_one++;
                        }
                    }
                    if (inter.is_n_fold(3)) {
                        draws++;
                    }
                    inter.reset_engines();
                    inter.history.clear();
                }


                inter.process();
            }
        }


    }
    for (auto &inter : interfaces) {
        inter.terminate_engines();
    }
    int status;
    for (int p = 0; p < num_matches; ++p) {
        for (int k = 0; k < numEngines; ++k) {
            wait(&status);
        }
    }
    std::cout << "Finished the match" << std::endl;
}

void Match::set_arg1(std::string arg) {
    arg1 = arg;
}

void Match::set_arg2(std::string arg) {
    arg2=arg;
}


void Match::set_time(int time_one, int time_two) {
    times.first=time_one;
    times.second=time_two;
}
