//
// Created by root on 03.02.21.
//

#include "Generator.h"

void Generator::set_buffer_clear_count(size_t count) {
    buffer_clear_count = count;
}

void Generator::load_filter() {
    std::ifstream stream(output + std::string(".filter"), std::ios::binary);

    if (!stream) {
        std::cout << "There was no filter file" << std::endl;
        past_uniq_counter=0;
        pos_counter =0;
        return;
    }
    stream.read((char*)&past_uniq_counter,sizeof(size_t));
    stream.read((char*)&pos_counter,sizeof(size_t));
    const size_t size = num_bits / 8 + 1;
    stream.read((char *) (bit_set), sizeof(uint8_t) * size);
    stream.close();
}

void Generator::save_filter(size_t uniq_pos_seen,size_t pos_seen) {
    std::ofstream stream(output + std::string(".filter"), std::ios::binary);
    const size_t size = num_bits / 8 + 1;
    stream.write((char*)&uniq_pos_seen,sizeof(size_t));
    stream.write((char*)&pos_seen,sizeof(size_t));
    stream.write((char *) (bit_set), sizeof(uint8_t) * size);
    stream.close();
}


void Generator::set(size_t index) {
    const size_t part_index = index / 8u;
    const size_t sub_index = index % 8u;
    const uint8_t maske = (1u << sub_index) | bit_set[part_index];
    bit_set[part_index] = maske;
}

bool Generator::get(size_t index) {
    const size_t part_index = index / 8u;
    const size_t sub_index = index % 8u;
    const uint8_t maske = 1u << sub_index;
    return ((bit_set[part_index] & maske) == maske);
}

void Generator::insert(Sample sample) {
    uint64_t hash_val = hash(sample);

    auto hash1 = static_cast<uint32_t>(hash_val);
    auto hash2 = static_cast<uint32_t>(hash_val >> 32);

    for (uint32_t k = 0; k < num_hashes; ++k) {
        uint32_t current_hash = hash1 + hash2 * k;
        size_t index = current_hash % num_bits;
        set(index);
    }
}

bool Generator::has(const Sample &other) {
    uint64_t hash_val = hash(other);
    //extracing lower and upper 32 bits
    auto hash1 = static_cast<uint32_t>(hash_val);
    auto hash2 = static_cast<uint32_t>(hash_val >> 32);
    for (uint32_t k = 0; k < num_hashes; ++k) {
        uint32_t current_hash = hash1 + hash2 * k;
        size_t index = current_hash % num_bits;
        if (get(index) == false)
            return false;
    }
    return true;
}

void Generator::set_hash_size(int size) {
    hash_size = size;
}

void Generator::set_time(int time) {
    time_control = time;
}

void Generator::print_stats() {

}

void Generator::startx() {
    //Positions to be saved to a file
    initialize();
    const size_t BUFFER_CAP = 1000000;
    std::cout << "Number of openings: " << openings.size() << std::endl;

    int *buffer_length = (int *) mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
                                      0);
    int *buffer_length_temp = (int *) mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
                                           0);
    Sample *buffer = (Sample *) mmap(NULL, sizeof(Sample) * BUFFER_CAP, PROT_READ | PROT_WRITE,
                                     MAP_SHARED | MAP_ANONYMOUS, -1,
                                     0);
    //temporary buffer to see if the bloom-filter is working
    Sample *check_buffer = (Sample *) mmap(NULL, sizeof(Sample) * BUFFER_CAP, PROT_READ | PROT_WRITE,
                                           MAP_SHARED | MAP_ANONYMOUS, -1,
                                           0);
    int *counter;
    int *error_counter;
    int *num_games;
    int *num_won;
    int *opening_counter;
    size_t *unique_pos_seen;
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
    bit_set = (uint8_t *) mmap(NULL, sizeof(uint8_t) * ((num_bits / 8) + 1), PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_ANONYMOUS, -1,
                               0);

    unique_pos_seen = (size_t *) mmap(NULL, sizeof(size_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1,
                                      0);

    for (auto i = 0; i < ((num_bits / 8) + 1); ++i) {
        bit_set[i] = 0;
    }
    load_filter();
    *unique_pos_seen = past_uniq_counter;
    *counter = pos_counter;
    *error_counter = 0;
    *num_won = 0;
    *num_games = 0;
    *buffer_length = 0;
    *buffer_length_temp = 0;
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
            const uint64_t seed = 13199312313ull + 23ull * i;
            initialize(seed);
            use_classical(true);
            TT.resize(hash_size);
            std::cout << "Init child: " << i << std::endl;
            //play a game and increment the opening-counter once more


            while (true) {
                pthread_mutex_lock(pmutex);
                if (*counter >= 10000000000ull) {
                    break;
                }
                Position opening = openings[*opening_counter];
                Board board;
                board = opening;
                pthread_mutex_unlock(pmutex);
                std::vector<Position> game;
                TT.clear();
                for (int move_count = 0; move_count < 600; ++move_count) {
                    MoveListe liste;
                    getMoves(board.getPosition(), liste);

                    if (liste.length() == 0) {
                        //end of the game, a player won
                        pthread_mutex_lock(pmutex);
                        (*num_won)++;
                        (*num_games)++;

                        for (auto &pos: game) {
                            Sample sample;
                            sample.position = pos;
                            sample.result = ((board.getMover() == BLACK) ? 1 : -1);
                            if (!has(sample)) {
                                buffer[(*buffer_length)++] = sample;
                                (*unique_pos_seen)++;
                                insert(sample);
                            }
                            check_buffer[(*buffer_length_temp)++] = sample;
                        }
                        (*counter) += game.size();
                        pthread_mutex_unlock(pmutex);
                        break;
                    }
                    uint32_t count;
                    count = std::count(game.begin(), game.end(), game.back());
                    if (count >= 3) {
                        pthread_mutex_lock(pmutex);
                        (*num_games)++;
                        //draw by repetition
                        for (auto &pos: game) {
                            Sample sample;
                            sample.position = pos;
                            sample.result = 0;
                            if (!has(sample)) {
                                buffer[(*buffer_length)++] = sample;
                                (*unique_pos_seen)++;
                                insert(sample);
                            }
                            check_buffer[(*buffer_length_temp)++] = sample;
                        }
                        (*counter) += game.size();
                        pthread_mutex_unlock(pmutex);
                        break;
                    }

                    Move best;
                    if (liste.length() == 1) {
                        best = liste[0];
                    } else {
                        searchValue(board, best, MAX_PLY, time_control, false);
                    }
                    board.makeMove(best);
                    /*    board.printBoard();
    */


                    auto num_pieces = Bits::pop_count(board.getPosition().BP | board.getPosition().WP);
                    uint32_t WP = board.getPosition().WP & (~board.getPosition().K);
                    uint32_t BP = board.getPosition().BP & (~board.getPosition().K);
                    uint32_t WK = board.getPosition().WP & (board.getPosition().K);
                    uint32_t BK = board.getPosition().BP & (board.getPosition().K);

                    if (num_pieces > 24 || std::abs(board.getPosition().color) != 1 || num_pieces == 0 ||
                        ((WP & BP) != 0) || ((WK & BK) != 0)) {
                        board.getPosition().printPosition();
                    }

                    game.emplace_back(board.getPosition());
                }
                pthread_mutex_lock(pmutex);
                if (*buffer_length >= buffer_clear_count) {
                    std::cout << "ClearedBuffer" << std::endl;
                    Utilities::write_to_binary<Sample>(buffer, buffer + *buffer_length, output,
                                                       std::ios::app);
                    Utilities::write_to_binary<Sample>(check_buffer, check_buffer + *buffer_length_temp,
                                                       output + ".check",
                                                       std::ios::app);
                    *buffer_length = 0;
                    *buffer_length_temp = 0;
                    save_filter(*unique_pos_seen,*counter);
                }
                std::cout << "Pos Seen: " << *counter << std::endl;
                std::cout << "Unique-Pos-Seen: " << *unique_pos_seen << std::endl;
                std::cout << "Ratio: " << (double) (*unique_pos_seen) / (double) (*counter) << std::endl;
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
            std::exit(1);
        }
    }

    if (id > 0) {
        //main_process
        int status = 0;
        while ((id = wait(&status)) > 0) {
            printf("Exit status of %d was %d (%s)\n", (int) id, status,
                   (status > 0) ? "accept" : "reject");
            std::exit(-1);
        }
    }




    /* Clean up. */
    pthread_mutex_destroy(pmutex);
    pthread_mutexattr_destroy(&attrmutex);
}

void Generator::set_parallelism(size_t threads) {
    parallelism = threads;
}

void Generator::create_filter_file(std::string input) {
    //make sure to "null" the bloom-filter before returning from this
    //function


}

