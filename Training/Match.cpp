
// Created by robin on 7/26/18.
//

#include "Match.h"

bool Interface::is_n_fold(int n) {
  if (history.empty())
    return false;

  const Position other = history.back();
  return std::count(history.begin(), history.end(), other) >= n;
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

void Engine::setTime(int time) { time_move = time; }

void Engine::setHashSize(int hash) { hash_size = hash; }

std::string Engine::readPipe() {
  std::string message;
  char c;
  int result;
  do {
    result = read(engineRead, (char *)(&c), sizeof(char));
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
  write(engineWrite, (char *)&ex_message.front(),
        sizeof(char) * ex_message.size());
}

// CheckerBoard's maxtime_for_non_incremental_tc (eygilbert/CheckerBoard,
// CheckerBoard.c): 'remaining' is the bank after crediting this move's level.
static double cb_maxtime(double remaining, double increment) {
  if (remaining < 0.4 * increment)
    return 0.4 * increment;
  if (remaining > increment)
    return increment + (remaining - increment) / 2.5;
  return remaining;
}

Move Engine::search() {
  if (state == State::Game_Ready) {
    if (!waiting_response) {
      int maxtime = time_move;
      if (use_bank) {
        bank_ms += time_move;
        maxtime = std::max(1, (int)std::lround(cb_maxtime(bank_ms, time_move)));
      }
      sent_maxtime = maxtime;
      writeMessage("search");
      writeMessage(std::to_string(maxtime));
      search_start = std::chrono::steady_clock::now();
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
      const double elapsed_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - search_start)
              .count();
      if (use_bank)
        bank_ms -= elapsed_ms;
      searches++;
      used_ms_sum += elapsed_ms;
      maxtime_ms_sum += sent_maxtime;
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
      writeMessage(pos.get_fen_string());
    }
    waiting_response = true;
    auto answer = readPipe();
    if (answer == "game_ready") {
      state = State::Game_Ready;
      waiting_response = false;
      bank_ms = 0.0; // CheckerBoard resets the clocks at every game
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
  if (is_n_fold(3) || history.size() >= 600)
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
    writeMessage(std::to_string(Bits::bitscan_foward(captures)));
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
  Move move;
  move = engines[first_mover].search();

  if (!move.is_empty() && Interface::is_legal_move(move)) {
    pos.make_move(move);
    history.emplace_back(pos);
    engines[second_mover].new_move(move);
    first_mover = second_mover;
  } else if (!Interface::is_legal_move(move)) {

    std::cout << "Got an illegal move" << std::endl;
    if (move.is_empty()) {
      std::cout << "The move was empty" << std::endl;
    }
    pos.print_position();

    std::exit(EXIT_FAILURE);
  }
}

int Match::getMaxGames() { return maxGames; }

void Match::setMaxGames(int games) { this->maxGames = games; }

void Match::setTime(int time) { this->time = time; }

void Match::setOpeningBook(std::string book) {
  this->openingBook = book;
  load_book();
}

void Match::load_book() {
  std::ifstream stream(openingBook);
  if (!stream.good()) {
    std::cerr << "Could not load the openings from " << openingBook
              << std::endl;
    std::exit(-1);
  }
  positions.clear();
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (!line.empty())
      positions.emplace_back(line);
  }
  if (positions.size() < 2) {
    std::cerr << "Opening book " << openingBook << " has too few positions"
              << std::endl;
    std::exit(-1);
  }
}

void Match::set_sprt(double elo0, double elo1, double alpha, double beta) {
  sprt_enabled = true;
  this->elo0 = elo0;
  this->elo1 = elo1;
  this->alpha = alpha;
  this->beta = beta;
}

void Match::print_status(int games) {
  double elo, err;
  penta.elo(elo, err);
  printf("games %-6d W-L-D %d-%d-%d  penta [%llu %llu %llu %llu %llu]  "
         "Elo %+.1f +- %.1f",
         games, wins_one, wins_two, draws, (unsigned long long)penta.n[0],
         (unsigned long long)penta.n[1], (unsigned long long)penta.n[2],
         (unsigned long long)penta.n[3], (unsigned long long)penta.n[4], elo,
         err);
  if (sprt_enabled) {
    printf("  LLR %.2f [%.2f, %.2f]", penta.llr(elo0, elo1),
           Sprt::lower_bound(alpha, beta), Sprt::upper_bound(alpha, beta));
  }
  printf("\n");
  fflush(stdout);
}

const std::string &Match::getOpeningBook() { return openingBook; }

void Match::setNumThreads(int threads) { this->threads = threads; }

int Match::getNumThreads() { return threads; }

void Match::setHashSize(int hash) { this->hash_size = hash; }

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

std::string Match::get_start_pos(size_t &pos_counter) {
  if (pos_counter >= positions.size() - 1)
    pos_counter = 0u;

  return positions[pos_counter++];
}

void Match::start() {
  // this probably needs to be reworked
  system("echo ' \\e[1;31m Engine Match \\e[0m' ");
  std::cout << "Engine1: " << first << std::endl;
  std::cout << "Engine2: " << second << std::endl;

  const int numEngines = 2;
  const int num_matches = this->threads;

  std::vector<Interface> interfaces;
  std::cout << "OpeningBook: " << openingBook << std::endl;
  std::cout << "Number of Positions: " << positions.size() << std::endl;
  std::vector<std::string> engine_paths{first, second};

  int mainPipe[num_matches][numEngines][2];
  int enginePipe[num_matches][numEngines][2];
  int game_count = -num_matches;

  const size_t num_local = positions.size() / num_matches;
  for (int p = 0; p < num_matches; ++p) {
    Engine engine{first, Engine::State::Idle, enginePipe[p][0][0],
                  mainPipe[p][0][1]};
    engine.setTime(times.first);
    engine.setHashSize(hash_size);
    engine.use_bank = time_bank;
    Engine engine2{second, Engine::State::Idle, enginePipe[p][1][0],
                   mainPipe[p][1][1]};
    engine2.setTime(times.second);
    engine2.setHashSize(hash_size);
    engine2.use_bank = time_bank;
    interfaces.emplace_back(Interface{engine, engine2});
    interfaces[p].pos_counter = num_local * p;
  }
  if (use_book_seed) {
    std::mt19937_64 book_rng(book_seed);
    for (auto &inter : interfaces)
      inter.pos_counter = book_rng() % (positions.size() - 1);
  }
  auto &first = interfaces.front();
  std::cout << "Time: " << times.first << " " << times.second
            << (time_bank ? " (CheckerBoard time bank)" : " (fixed per move)")
            << std::endl;

  pid_t pid;
  int finished_games = 0;
  int sprt_result = 0; // -1: H0 accepted, 1: H1 accepted
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
        auto argument = (i == 0) ? arg1 : arg2;
        if (argument.empty()) {

          execlp(command.c_str(), e.c_str(), (char *)NULL);
        } else {

          execlp(command.c_str(), e.c_str(), argument.c_str(), (char *)NULL);
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
    if (sprt_enabled) {
      printf("SPRT: elo0 %.2f elo1 %.2f alpha %.3f beta %.3f (logistic Elo, "
             "pentanomial)\n",
             elo0, elo1, alpha, beta);
    }
    const double llr_lower = Sprt::lower_bound(alpha, beta);
    const double llr_upper = Sprt::upper_bound(alpha, beta);
    while (game_count < maxGames && sprt_result == 0) {
      for (int k = 0; k < interfaces.size(); ++k) {
        Interface &inter = interfaces[k];
        if (inter.pos.is_empty()) {
          if (inter.first_game) {
            inter.first_mover = 0;
            std::string fen_string = get_start_pos(inter.pos_counter);
            // std::cout<<fen_string<<std::endl;
            inter.start_pos = Position::pos_from_fen(fen_string);
            inter.first_game = false;
          } else {
            inter.first_mover = 1;
            inter.first_game = true;
          }
          inter.pos = inter.start_pos;
        }

        if (inter.is_terminal_state()) {
          game_count++;
          finished_games++;
          MoveListe liste;
          get_moves(inter.pos, liste);
          int result; // engine 1's result in half points
          if (liste.length() == 0) {
            // the side to move has lost
            if (inter.first_mover == 0) {
              wins_two++;
              result = 0;
            } else {
              wins_one++;
              result = 2;
            }
          } else {
            draws++;
            result = 1;
          }

          // first_game is false while the first game of a pair is running
          // and true again once the colour-swapped second game has started
          if (!inter.first_game) {
            inter.pending_result = result;
          } else if (inter.pending_result >= 0) {
            penta.add(inter.pending_result + result);
            inter.pending_result = -1;
            print_status(finished_games);
            if (sprt_enabled) {
              const double llr = penta.llr(elo0, elo1);
              if (llr >= llr_upper)
                sprt_result = 1;
              else if (llr <= llr_lower)
                sprt_result = -1;
            }
          }

          inter.reset_engines();
          inter.history.clear();
          continue;
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
  if (pid > 0) {
    for (int e = 0; e < 2; ++e) {
      uint64_t n = 0;
      double used = 0.0, sent = 0.0;
      for (auto &inter : interfaces) {
        n += inter.engines[e].searches;
        used += inter.engines[e].used_ms_sum;
        sent += inter.engines[e].maxtime_ms_sum;
      }
      if (n > 0)
        printf("Engine%d time: %llu moves, avg used %.2f ms, avg maxtime sent "
               "%.2f ms\n",
               e + 1, (unsigned long long)n, used / n, sent / n);
    }
    std::cout << "Final: ";
    print_status(finished_games);
    if (sprt_enabled) {
      const double llr = penta.llr(elo0, elo1);
      if (sprt_result == 1)
        printf("SPRT: H1 accepted (elo0=%.2f elo1=%.2f), LLR %.2f\n", elo0,
               elo1, llr);
      else if (sprt_result == -1)
        printf("SPRT: H0 accepted (elo0=%.2f elo1=%.2f), LLR %.2f\n", elo0,
               elo1, llr);
      else
        printf("SPRT: inconclusive after %d games, LLR %.2f\n",
               finished_games, llr);
    }
  }
  std::cout << "Finished the match" << std::endl;
}

void Match::set_arg1(std::string arg) { arg1 = arg; }

void Match::set_arg2(std::string arg) { arg2 = arg; }

void Match::set_time(int time_one, int time_two) {
  times.first = time_one;
  times.second = time_two;
}
