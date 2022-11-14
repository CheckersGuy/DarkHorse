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

void Generator::set_max_position(size_t max) {
    max_positions = max;
}

void Generator::set_piece_limit(size_t num_pieces) {
    piece_lim = num_pieces;
}

void Generator::set_book(std::string book){
    std::string opening_path{"../Training/Positions/"};
    opening_path += book;

    std::ifstream stream(opening_path, std::ios::binary);
    std::istream_iterator<Position> begin(stream);
    std::istream_iterator<Position> end;
    std::copy(begin, end, std::back_inserter(openings));
    std::cout << "Size: " << openings.size() << std::endl;
}

void Generator::set_output(std::string output){
    this->output = output;
}

void Generator::set_max_games(size_t max_games){
    this->max_games =max_games;
}

void Generator::set_network(std::string net){
    net_file =net;
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
            network.addLayer(Layer{120, 1024});
            network.addLayer(Layer{1024, 8});
            network.addLayer(Layer{8, 32});
            network.addLayer(Layer{32, 1});
            network.load(net_file);
			std::cout<<net_file<<std::endl;
            network.init();

            TT.resize(hash_size);
            std::cout << "Init child: " << i << std::endl;
            //play a game and increment the opening-counter once more
            const uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() ^ getpid();
            std::mt19937_64 generator(seed);
            while (!stop) {
                Game game;
                std::uniform_int_distribution<size_t>distrib(0,openings.size());
                Position opening;
                const size_t rand_index = distrib(generator);
                opening = openings[rand_index];
                Board board;
                board = opening;
                TT.clear();
                Zobrist::init_zobrist_keys(seed);
                bool sucess;
                for (int move_count = 0; move_count < 600; ++move_count) {
                    MoveListe liste;
                    get_moves(board.get_position(), liste);
                    sucess =game.add_position(board.get_position());

                    if(!sucess){
                        break;
                    }


                    const Position p = board.get_position();
                    if (Bits::pop_count(p.BP | p.WP) <= piece_lim && (!p.has_jumps())) {
                         pthread_mutex_lock(pmutex);
                         (*num_games)++;
                          pthread_mutex_unlock(pmutex);
                        break;
                    }
                    uint32_t count;
					std::vector<Position>positions;
					game.extract_positions(std::back_inserter(positions));
                    count = std::count(positions.begin(), positions.end(), positions.back());
					if (liste.length() == 0) {
                        //end of the game, a player won
                        pthread_mutex_lock(pmutex);
                        (*num_won)++;
                        (*num_games)++;
                        pthread_mutex_unlock(pmutex);
                        break;
                    } else if (count >= 3) {
                        pthread_mutex_lock(pmutex);
                        (*num_games)++;
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
                if(game.indices.size()>0&& sucess){
                    game_buffer.emplace_back(game);
                }
                 
                if (game_buffer.size() >= buffer_clear_count) {

                    for (auto g: game_buffer) {
                        out_stream << g;
                    }
                    game_buffer.clear();
                pthread_mutex_lock(pmutex);
                const int current_games = (*num_games);
                std::cout<<current_games<<std::endl;
                pthread_mutex_unlock(pmutex);
                if(current_games>=max_games){
                    break;
                }
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
