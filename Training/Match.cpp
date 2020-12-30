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
            Logger::get_instance() << "Game ready" << "\n";
        }
    }
}


void Engine::update() {
    if (state == State::Update) {
        auto answer = readPipe();
        if (answer == "update_ready") {
            waiting_response = false;
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
    if (is_n_fold(3) || history.size() >= 400)
        return true;

    MoveListe liste;
    getMoves(pos, liste);
    return liste.length() == 0;

}

void Engine::new_move(Move move) {
    writeMessage("new_move");
    writeMessage(std::to_string(move.getFromIndex()));
    writeMessage(std::to_string(move.getToIndex()));
    uint32_t captures = move.captures;
    while (captures) {
        writeMessage(std::to_string(__tzcnt_u32(captures)));
        captures &= captures - 1u;
    }
    writeMessage("end_move");
}


void Interface::process() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto &logger = Logger::get_instance();
    for (auto &engine : engines) {
        engine.initEngine();
        engine.newGame(pos);
        engine.update();
    }
    const int second_mover = (first_mover == 0) ? 1 : 0;

    //checking if there is only one possible move
    MoveListe liste;
    getMoves(pos, liste);

    Move move{};

    move = engines[first_mover].search();


    if (!move.isEmpty()) {

        if (!Interface::is_legal_move(move)) {
            logger << "Error: Illegal move" << "\n";
            logger << "From: " << move.getFromIndex() << "\n";
            logger << "To: " << move.getToIndex() << "\n";
            std::exit(EXIT_FAILURE);
        }
        pos.makeMove(move);
        history.emplace_back(pos);
        engines[second_mover].new_move(move);
        first_mover = second_mover;


        logger << pos.position_str() << "\n";
        logger << getPositionString(pos) << "\n";
        if (liste.length() == 1)
            logger << "Easy_move" << "\n";
        logger << "Move: [" << std::to_string(move.getFromIndex()) << ", "
               << std::to_string(move.getToIndex()) << "]"
               << "\n";
        logger << "Engine " << first_mover << ": " << engines[first_mover].engine_name << "\n";
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
    for (auto &engine : engines) {
        engine.state = Engine::State::Idle;
    }
    pos = Position{};
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
    //this probably needs to be reworked
    HyperLog<Position, 14, TrainHasher> counter;
    system("echo ' \\e[1;31m Engine Match \\e[0m' ");
    Zobrist::initializeZobrisKeys();
    const int numEngines = 2;
    const int num_matches = this->threads;

    if (num_matches == 1) {
        Logger::get_instance().turn_on();
    }


    std::vector<Interface> interfaces;

    Training::TrainData positions;
    std::ifstream in(openingBook);
    positions.ParseFromIstream(&in);
    in.close();

    std::cout << "OpeningBook: " << openingBook << std::endl;
    std::cout << "Number of Positions: " << positions.positions_size() << std::endl;
    std::cout << "Output_File: " << output_file << std::endl;
    std::vector<std::string> engine_paths{first, second};


    int mainPipe[num_matches][numEngines][2];
    int enginePipe[num_matches][numEngines][2];
    int game_count = -num_matches;

    for (int p = 0; p < num_matches; ++p) {
        Engine engine{first, Engine::State::Idle, enginePipe[p][0][0], mainPipe[p][0][1]};
        engine.setTime(time);
        engine.setHashSize(hash_size);
        Engine engine2{second, Engine::State::Idle, enginePipe[p][1][0], mainPipe[p][1][1]};
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
                const std::string e = "../Training/Engines/" + engine_paths[i];
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
                fcntl(enginePipe[p][k][0], F_SETFL, O_NONBLOCK | O_RDONLY);
            }
        }
        printf("%-5s %-5s %-5s %-5s \n", "Wins_one", "Wins_two", "Draws", "Uniq_Counter");
        printf("%-5d %-5d %-5d %-5d", wins_one, wins_two, draws, (int) counter.get_count());
        printf("\r");
        printf("\r");
        int start_index = 0;
        auto &logger = Logger::get_instance();

        while (game_count < maxGames) {
            for (auto &inter : interfaces) {
                if (inter.pos.isEmpty()) {
                    if (!inter.played_reverse) {
                        start_index = (start_index >= positions.positions().size() - 1) ? 0 : start_index + 1;
                        const Training::Position &temp = positions.positions(start_index);
                        Position pos;
                        pos.WP = temp.wp();
                        pos.BP = temp.bp();
                        pos.K = temp.k();
                        pos.color = (temp.mover() == Training::BLACK) ? BLACK : WHITE;
                        inter.start_pos=pos;
                        inter.pos = inter.start_pos;
                        inter.played_reverse = play_reverse;
                        inter.first_mover = 0;
                    } else {
                        inter.first_mover = 1;
                        inter.played_reverse = false;
                        inter.pos = inter.start_pos;
                    }
                }

                if (inter.is_terminal_state()) {
                    game_count++;
                    printf("%-5d %-5d %-5d %-5d", wins_one, wins_two, draws, (int) counter.get_count());
                    std::cout << std::endl;
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
                            p.key = Zobrist::generateKey(p);
                            addPosition(p, result);
                            counter.insert(p);
                        }
                    }
                    if (inter.is_n_fold(3)) {
                        logger << "Repetition" << "\n";
                        draws++;
                        for (Position &p : inter.history) {
                            p.key = Zobrist::generateKey(p);
                            addPosition(p, Training::DRAW);
                            counter.insert(p);
                        }
                    }
                    if (inter.history.size() >= 400) {
                        logger << "Reached max move" << "\n";
                    }
                    logger << inter.pos.position_str() << "\n";

                    logger << "End of the game with index: " << start_index << "\n";
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
    Logger::get_instance().turn_off();
    std::ofstream stream(this->output_file);
    if (!stream.good())
        std::cerr << "Could not save the training-data" << std::endl;
    data.SerializeToOstream(&stream);
    stream.close();
    std::cout << "Finished the match" << std::endl;
}
