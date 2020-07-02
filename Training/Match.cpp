//
// Created by robin on 7/26/18.
//

#include "Match.h"


bool Interface::is_n_fold(int n) {
    if (history.empty())
        return false;

    const Position other = history.back();
    auto count = std::count_if(history.begin(), history.end(), [&other](Position &p) {
        return p == other;
    });
    return count >= n;
}


bool Interface::is_legal_move(Move move) {
    MoveListe liste;
    getMoves(pos, liste);
    Position check_pos = pos;
    check_pos.makeMove(move);
    for (auto m : liste.liste) {
        Position copy = pos;
        copy.makeMove(m);
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

std::optional<Move> Engine::search() {
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

            return  std::make_optional(move);
        }
    }
    return std::nullopt;
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
            Logger::get_instance() << "Game ready" << "\n";
        }
    }
}


void Engine::update() {
    if (state == State::Update) {
        auto answer = readPipe();
        if (answer == "update_ready") {
            state = State::Game_Ready;
            Logger::get_instance() << "Updated" << "\n";
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
            Logger::get_instance() << "Init ready" << "\n";
        }
    }

}

bool Interface::is_terminal_state() {
    if (is_n_fold(3))
        return true;

    MoveListe liste;
    getMoves(pos, liste);
    return liste.length() == 0 || history.size() >= 400;

}


void Interface::process() {

    for (auto &engine : engines) {
        engine.initEngine();
        engine.newGame(pos);
        engine.update();
    }


    auto move = engines[first_mover].search();
    if (move.has_value()) {
        auto &logger = Logger::get_instance();
        const int second_mover = (first_mover == 0) ? 1 : 0;
        if (!Interface::is_legal_move(move.value())) {
            logger << "Error: Illegal move" << "\n";
            logger << "From: " << move->getFromIndex() << "\n";
            logger << "To: " << move->getToIndex() << "\n";
            std::exit(EXIT_FAILURE);
        }
        pos.makeMove(move.value());
        history.emplace_back(pos);
        engines[second_mover].state = Engine::State::Update;
        engines[second_mover].writeMessage("new_move");
        engines[second_mover].writeMessage(std::to_string(__tzcnt_u32(move.value().from)));
        engines[second_mover].writeMessage(std::to_string(__tzcnt_u32(move.value().to)));
        uint32_t captures = move->captures;
        while (captures) {
            engines[second_mover].writeMessage(std::to_string(__tzcnt_u32(captures)));
            captures &= captures - 1u;
        }
        engines[second_mover].writeMessage("end_move");

        first_mover = second_mover;
        logger << pos.position_str() << "\n";
        logger << getPositionString(pos) << "\n";
        logger << "\n";
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

std::string &Match::getOpeningBook() {
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
    first_mover = 0;
    for (auto &engine : engines) {
        engine.state = Engine::State::Idle;
    }
}

std::string Match::get_output_file() {
    return output_file;
}


void Match::addPosition(Position p, Training::Result result) {
    auto pos = data.add_positions();
    pos->set_k(p.K);
    pos->set_bp(p.BP);
    pos->set_wp(p.WP);
    pos->set_mover((p.color == BLACK) ? Training::BLACK : Training::WHITE);
    pos->set_result(result);
}

void Match::set_play_reverse(bool flag) {
    play_reverse = flag;
}

void Interface::terminate_engines() {
    for (auto &engine : engines) {
        engine.writeMessage("terminate");
    }
}


void Match::start() {
    system("echo ' \\e[1;31m Engine Match \\e[0m' ");
    Zobrist::initializeZobrisKeys();
    const int numEngines = 2;
    const int num_matches = this->threads;

    if (num_matches == 1) {
        Logger::get_instance().turn_on();
    }


    std::vector<Interface> interfaces;
    std::vector<Position> positions;
    Utilities::loadPositions(positions, openingBook);
    std::vector<std::string> engine_paths{first, second};


    int mainPipe[num_matches][numEngines][2];
    int enginePipe[num_matches][numEngines][2];
    int eng_info_pipe[num_matches][numEngines][2];

    int start_index = 0;
    int game_count = -num_matches;

    for (int p = 0; p < num_matches; ++p) {
        Engine engine{Engine::State::Idle, enginePipe[p][0][0], mainPipe[p][0][1]};
        engine.setTime(time);
        engine.setHashSize(hash_size);
        Engine engine2{Engine::State::Idle, enginePipe[p][1][0], mainPipe[p][1][1]};
        engine2.setTime(time);
        engine2.setHashSize(hash_size);
        interfaces.emplace_back(Interface{engine, engine2});
    }


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
                dup2(eng_info_pipe[p][i][1], STDERR_FILENO);
                const std::string e = "../Engines/" + engine_paths[i];
                const std::string command = "./" + e;
                Logger::get_instance() << command << "\n";
                execlp(command.c_str(), e.c_str(), NULL);
                exit(EXIT_SUCCESS);
            }
        }
    }
    if (pid > 0) {
        for (int p = 0; p < num_matches; ++p) {
            for (int k = 0; k < numEngines; ++k) {
                close(mainPipe[p][k][0]);
                close(enginePipe[p][k][1]);
                close(eng_info_pipe[p][k][1]);
                fcntl(enginePipe[p][k][0], F_SETFL, O_NONBLOCK | O_RDONLY);
                fcntl(eng_info_pipe[p][k][0], F_SETFL, O_NONBLOCK | O_RDONLY);
            }
        }
        printf("%-5s %-5s %-5s \n", "Wins_one", "Wins_two", "Draws");
        while (game_count < maxGames) {
            for (auto &inter : interfaces) {
                auto &logger = Logger::get_instance();
                if (inter.is_terminal_state()) {
                    //need to set up a new position
                    logger << "Start of the game" << "\n";
                    if (!inter.played_reverse || !play_reverse) {
                        start_index = (start_index >= positions.size()) ? 0 : start_index + 1;
                        Position &pos = positions[start_index];
                        inter.pos = pos;
                        inter.played_reverse = !inter.played_reverse;
                        inter.first_mover = 0;
                    } else {
                        inter.first_mover = 1;
                        inter.played_reverse = !inter.played_reverse;
                        inter.pos = inter.history.front();
                    }

                    logger << inter.pos.position_str() << "\n";
                    logger << "First_Mover: " << inter.first_mover << "\n";

                    printf("%-5d %-5d %-5d", wins_one, wins_two, draws);
                    printf("\r");
                    std::cout.flush();
                    inter.history.clear();
                    inter.reset_engines();
                    game_count++;
                }

                inter.process();

                MoveListe liste;
                getMoves(inter.pos, liste);
                if (liste.length() == 0) {
                    logger << "End game" << "\n";
                    if (inter.first_mover == 0) {
                        wins_two++;
                    } else {
                        wins_one++;
                    }
                    Training::Result result;
                    result = (inter.pos.color == BLACK) ? Training::WHITE_WON : Training::BLACK_WON;
                    for (Position &p : inter.history) {
                        addPosition(p, result);
                    }
                }
                if (inter.is_n_fold(3)) {
                    logger << "Repetition" << "\n";
                    draws++;
                    for (Position &p : inter.history) {
                        addPosition(p, Training::DRAW);
                    }
                }
                if (inter.history.size() >= 400) {
                    logger << "Reached max move" << "\n";
                }
            }
            if (game_count >= maxGames)
                break;
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
    Logger::get_instance().turn_off();
    std::ofstream stream("output_file", std::ios::binary);
    data.SerializeToOstream(&stream);
    stream.close();
    std::cout << "Finished the match" << std::endl;
}
