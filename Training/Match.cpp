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


bool Interface::isLegalMove(Move move) {
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

            return move;
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
            std::cerr << "Game ready" << std::endl;
        }
    }
}


void Engine::update() {
    if (state == State::Update) {
        auto answer = readPipe();
        if (answer == "update_ready") {
            state = State::Game_Ready;
            std::cerr << "Updated" << std::endl;
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
            std::cerr << "Init ready" << std::endl;
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
        const int second_mover = (first_mover == 0) ? 1 : 0;
        if (!Interface::isLegalMove(move.value())) {
            std::cerr << "Illegal move" << "\n";
            std::cerr << "From: " << move->getFromIndex() << "\n";
            std::cerr << "To: " << move->getToIndex() << "\n";
            exit(EXIT_FAILURE);
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
        std::cerr << pos.position_str() << "\n";
        std::cerr << getPositionString(pos) << "\n";
        std::cerr << "\n";
    }
}


Match::~Match() {
    logger.close();
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

void Match::start() {
    bool reverse = false;
    system("echo ' \\e[1;31m Engine Match \\e[0m' ");

    auto previous_buffer_cerr = std::cerr.rdbuf();
    std::cerr.rdbuf(logger.rdbuf());
    Zobrist::initializeZobrisKeys();
    const int numEngines = 2;
    int mainPipe[numEngines][2];
    int enginePipe[numEngines][2];
    int eng_info_pipe[numEngines][2];

    int start_index = 0;
    int game_count = 0;

    Engine engine{Engine::State::Idle, enginePipe[0][0], mainPipe[0][1]};
    engine.setTime(time);
    engine.setHashSize(hash_size);
    Engine engine2{Engine::State::Idle, enginePipe[1][0], mainPipe[1][1]};
    engine2.setTime(time);
    engine2.setHashSize(hash_size);
    Interface inter{engine, engine2};

    std::vector<Position> positions;
    Utilities::loadPositions(positions, openingBook);
    std::vector<std::string> engine_paths{first, second};


    pid_t pid;
    for (auto i = 0; i < numEngines; ++i) {
        pipe(mainPipe[i]);
        pipe(enginePipe[i]);
        pid = fork();
        if (pid < 0) {
            std::cerr << "Error" << std::endl;
            exit(EXIT_FAILURE);
        } else if (pid == 0) {
            dup2(mainPipe[i][0], STDIN_FILENO);
            dup2(enginePipe[i][1], STDOUT_FILENO);
            dup2(eng_info_pipe[i][1], STDERR_FILENO);
            const std::string e = "../Engines/" + engine_paths[i];
            const std::string command = "./" + e;
            std::cerr << command << std::endl;
            execlp(command.c_str(), e.c_str(), NULL);
            exit(EXIT_SUCCESS);
        }
    }
    if (pid > 0) {
        for (int k = 0; k < numEngines; ++k) {
            close(mainPipe[k][0]);
            close(enginePipe[k][1]);
            close(eng_info_pipe[k][1]);
            fcntl(enginePipe[k][0], F_SETFL, O_NONBLOCK | O_RDONLY);
            fcntl(eng_info_pipe[k][0], F_SETFL, O_NONBLOCK | O_RDONLY);
        }
        printf("%-5s %-5s %-5s \n", "Wins_one", "Wins_two", "Draws");
        while (game_count < maxGames) {

            if (inter.is_terminal_state()) {
                //need to set up a new position
                std::cerr << "Start of the game" << std::endl;
                Position &pos = positions[start_index];
                std::cerr << pos.position_str() << std::endl;
                inter.pos = pos;
                inter.first_mover = 0;
                inter.history.clear();
                start_index = (start_index >= positions.size()) ? 0 : start_index + 1;
                inter.reset_engines();
                printf("%-5d %-5d %-5d", wins_one, wins_two, draws);
                printf("\r");
            }
            inter.process();

            MoveListe liste;
            getMoves(inter.pos, liste);
            if (liste.length() == 0) {
                std::cerr << "End game" << std::endl;
                if (inter.first_mover == 0) {
                    wins_two++;
                } else {
                    wins_one++;
                }
            }
            if (inter.is_n_fold(3)) {
                std::cerr << "Repetition" << std::endl;
                draws++;
            }
            if (inter.history.size() >= 400) {
                std::cerr << "Reached max move" << std::endl;
            }

        }


    }


    int status;
    for (int k = 0; k < numEngines; ++k) {
        wait(&status);
    }
    std::cerr.rdbuf(previous_buffer_cerr);
}
