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

void Generator::start() {
    //Positions to be saved to a file
    initialize();
    std::cout << "Number of openings: " << openings.size() << std::endl;

    int *num_games;
    int *num_won;
    num_won = (int *) mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
                           0);
    num_games = (int *) mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
                             0);


    *num_won = 0;
    *num_games = 0;
    pmutex = NULL;
    pthread_mutexattr_t attrmutex;
    pthread_mutexattr_setpshared(&attrmutex, PTHREAD_PROCESS_SHARED);

    pmutex = (pthread_mutex_t *) mmap(NULL, sizeof(pthread_mutex_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,
                                      -1, 0);

    pthread_mutex_init(pmutex, &attrmutex);
    bool stop = false;

    pid_t id;
    for (auto i = 0; i < parallelism; ++i) {
        id = fork();
        if (id < 0) {
            std::cerr << "Could not fork the main process" << std::endl;
            std::exit(-1);
        }
        if (id == 0) {
            const std::string local_file = output + ".temp" + std::to_string(i);
            std::ofstream out_stream("../Training/TrainData/" + local_file);
            //child takes a position and generates games
            std::vector<Game> game_buffer;
            use_classical(true);

            TT.resize(hash_size);
            std::cout << "Init child: " << i << std::endl;
            //play a game and increment the opening-counter once more


            while (!stop) {
                const uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() ^ getpid();
                Game game;

                std::mt19937_64 generator(seed);
                std::uniform_int_distribution<size_t>distrib(0,openings.size());
                const size_t rand_index = distrib(generator);
                Position opening = openings[rand_index];
                //std::cout<<"Opening counter: "<<rand_index<<std::endl;
                Board board;
                board = opening;
                TT.clear();
                Zobrist::init_zobrist_keys(seed);
                
                for (int move_count = 0; move_count < 600; ++move_count) {
                    MoveListe liste;
                    get_moves(board.get_position(), liste);
                    game.add_position(board.get_position());
                    const Position p = board.get_position();
                    if (Bits::pop_count(p.BP | p.WP) <= piece_lim && (!p.has_jumps())) {
                        game_buffer.emplace_back(game);
                        break;
                    }
                    uint32_t count;
                    count = std::count(game.begin(), game.end(), game.get_last_position());
                    if (liste.length() == 0) {
                        //end of the game, a player won
                        pthread_mutex_lock(pmutex);
                        (*num_won)++;
                        (*num_games)++;
                        game_buffer.emplace_back(game);
                        pthread_mutex_unlock(pmutex);
                        break;
                    } else if (count >= 3) {
                        pthread_mutex_lock(pmutex);
                        (*num_games)++;
                        game_buffer.emplace_back(game);
                        pthread_mutex_unlock(pmutex);
                        break;
                    }
                    if (liste.length() == 1) {
                        board.play_move(liste[0]);
                    } else {
                        Move best;
                        auto value = searchValue(board, best, MAX_PLY, time_control, false,std::cout);
                        board.play_move(best);
                    }
                

                }

                if (game_buffer.size() >= 100) {
                    //clearing the buffer after 100 games have been accumulated
                    for (auto &g: game_buffer) {
                        out_stream << g;
                    }
                    game_buffer.clear();
                }
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


