//
// Created by root on 03.02.21.
//

#include "Generator.h"

void Generator::set_buffer_clear_count(size_t count) {
    buffer_clear_count = count;
}

void Generator::set_hash_size(int size) {
    hash_size = size;
}

void Generator::set_time(int time) {
    time_control = time;
}

uint64_t Generator::get_shared_random_number() {

    //locking the acess to the generator
    pthread_mutex_lock(pmutex);


    pthread_mutex_unlock(pmutex);

}

void Generator::set_max_position(size_t max) {
    max_positions = max;
}

void Generator::set_piece_limit(size_t num_pieces) {
    piece_lim = num_pieces;
}

void Generator::startx() {
    //Positions to be saved to a file
    initialize();
    const size_t BUFFER_CAP = 1000000;
    std::cout << "Number of openings: " << openings.size() << std::endl;

    int *buffer_length = (int *) mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
                                      0);
    bool *stop = (bool *) mmap(NULL, sizeof(bool), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
                               0);
    Position *buffer = (Position *) mmap(NULL, sizeof(Position) * BUFFER_CAP, PROT_READ | PROT_WRITE,
                                         MAP_SHARED | MAP_ANONYMOUS, -1,
                                         0);
    //temporary buffer to see if the bloom-filter is working

    //shared random number generator
    random = (uint64_t *) mmap(NULL, sizeof(uint64_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
                               0);

    int *counter;
    int *error_counter;
    int *num_games;
    int *num_won;
    int *opening_counter;
    counter = (int *) mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
                           0);
    error_counter = (int *) mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
                                 0);
    num_won = (int *) mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
                           0);
    num_games = (int *) mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
                             0);

    opening_counter = (int *) mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
                                   0);
    *stop = false;
    *counter = 0;
    *error_counter = 0;
    *num_won = 0;
    *num_games = 0;
    *buffer_length = 0;
    *random = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    pmutex = NULL;
    pthread_mutexattr_t attrmutex;
    pthread_mutexattr_setpshared(&attrmutex, PTHREAD_PROCESS_SHARED);

    pmutex = (pthread_mutex_t *) mmap(NULL, sizeof(pthread_mutex_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,
                                      -1, 0);

    pthread_mutex_init(pmutex, &attrmutex);


    pid_t id;
    for (auto i = 0; i < parallelism; ++i) {
        id = fork();
        if (id < 0) {
            std::cerr << "Could not fork the main process" << std::endl;
            std::exit(-1);
        }
        if (id == 0) {
            //child takes a position and generates games
            initialize(13199312313ull + 12412312314ull * i);
            use_classical(true);

            TT.resize(hash_size);
            std::cout << "Init child: " << i << std::endl;
            //play a game and increment the opening-counter once more


            while (!(*stop)) {
                pthread_mutex_lock(pmutex);

                if (*counter >= max_positions) {
                    pthread_mutex_lock(pmutex);
                    *stop = true;
                    pthread_mutex_unlock(pmutex);
                    break;
                }

                Position opening = openings[*opening_counter];
                Board board;
                board = opening;
                pthread_mutex_unlock(pmutex);
                std::vector<Position> game;
                TT.clear();
                uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() ^ getpid();
                Zobrist::initializeZobrisKeys(seed);
                int move_count;
                for (move_count = 0; move_count < 600; ++move_count) {
                    MoveListe liste;
                    getMoves(board.getPosition(), liste);

                    //if the position happens to be an illegal position
                    //we skip this position
                    if (!board.getPosition().islegal()) {
                        pthread_mutex_lock(pmutex);
                        (*error_counter)++;
                        pthread_mutex_unlock(pmutex);
                        break;
                    }

                    game.emplace_back(board.getPosition());

                    //some form of adjudication trying


                    uint32_t count;
                    count = std::count(game.begin(), game.end(), game.back());
                    //experimental stuff below
                    if ((Bits::pop_count(board.getPosition().BP | board.getPosition().WP) <= piece_lim &&
                         !board.getPosition().hasJumps(BLACK) && !board.getPosition().hasJumps(WHITE))
                            ) {
                        for (auto &pos: game) {
                            buffer[(*buffer_length)++] = pos;
                            (*counter)++;
                        }
                        break;
                    }
                    if (liste.length() == 0) {
                        //end of the game, a player won
                        pthread_mutex_lock(pmutex);
                        (*num_won)++;
                        (*num_games)++;
                        for (auto &pos: game) {
                            buffer[(*buffer_length)++] = pos;
                            (*counter)++;
                        }
                        pthread_mutex_unlock(pmutex);
                        break;
                    } else if (count >= 3) {
                        pthread_mutex_lock(pmutex);
                        (*num_games)++;
                        //draw by repetition
                        for (auto &pos: game) {
                            buffer[(*buffer_length)++] = pos;
                            (*counter)++;
                        }
                        pthread_mutex_unlock(pmutex);
                        break;
                    }
                    if (liste.length() == 1) {
                        board.makeMove(liste[0]);
                    } else {
                        Move best;
                        auto value = searchValue(board, best, MAX_PLY, time_control, false);
                        board.makeMove(best);
                    }

                }
                pthread_mutex_lock(pmutex);
                if (*buffer_length >= buffer_clear_count) {
                    std::cout << "ClearedBuffer" << std::endl;
                    std::ofstream stream(output, std::ios::binary | std::ios::app);
                    std::copy(buffer, buffer + *buffer_length, std::ostream_iterator<Position>(stream));
                    *buffer_length = 0;
                }
                std::cout << "MoveCount: " << move_count << std::endl;
                std::cout << "Pos Seen: " << *counter << std::endl;
                std::cout << "Opening_Counter: " << *opening_counter << std::endl;
                std::cout << "Error_Counter: " << *error_counter << std::endl;
                std::cout << "WinRatio: " << (float) (*num_won) / (float) (*num_games) << std::endl;
                for (auto x = 0; x < 3; ++x) {
                    std::cout << "\n";
                }
                (*opening_counter)++;
                if (*opening_counter >= openings.size()) {
                    *opening_counter = 0;
                }
                pthread_mutex_unlock(pmutex);
            }
        }
    }

    if (id > 0) {
        //main_process
        int status = 0;
        while ((id = wait(&status)) > 0) {
            printf("Exit status of %d was %d (%s)\n", (int) id, status,
                   (status > 0) ? "accept" : "reject");
            if (status < 0) {

            }
        }
    }




    /* Clean up. */
    pthread_mutex_destroy(pmutex);
    pthread_mutexattr_destroy(&attrmutex);
}

void Generator::set_parallelism(size_t threads) {
    parallelism = threads;
}


