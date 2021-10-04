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

void Generator::print_stats() {

}

void Generator::set_max_position(size_t max) {
    max_positions = max;
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
    pthread_mutex_t *pmutex = NULL;
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
            const uint64_t seed = 13199312313ull + 12412312314ull * i;
            initialize(seed);
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
                initialize(std::chrono::high_resolution_clock::now().time_since_epoch().count());
                Position previous = opening;
                int move_count;
                for (move_count = 0; move_count < 400; ++move_count) {
                    MoveListe liste;
                    getMoves(board.getPosition(), liste);
                    game.emplace_back(board.getPosition());

                    //some form of adjudication trying


                    uint32_t count;
                    count = std::count(game.begin(), game.end(), game.back());
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
                    }else if (count >= 3) {
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

                    Move best;
                    auto t1 = std::chrono::high_resolution_clock::now();
                    auto value = searchValue(board, best, MAX_PLY, time_control, false);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    auto dur = t2-t1;
                    //std::cout<<"Took: "<<dur.count()/1000000<<std::endl;
                    board.makeMove(best);

                    if (Bits::pop_count(board.getPosition().BP | board.getPosition().WP) >
                        Bits::pop_count(previous.BP | previous.WP)) {
                        pthread_mutex_lock(pmutex);
                        *stop = true;
                        (*error_counter)++;
                        pthread_mutex_unlock(pmutex);
                    }
                    previous = board.getPosition();
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


