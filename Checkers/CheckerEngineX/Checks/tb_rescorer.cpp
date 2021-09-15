//
// Created by leagu on 13.09.2021.
//
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "egdb.h"
#include <vector>
#include "MGenerator.h"
#include "../../Training/Sample.h"
#include "../../Training/SampleFilter.h"
#ifdef ITALIAN_RULES
#define DB_PATH "c:/kr_english_wld"
#else
#define DB_PATH "c:/kr_english_wld"
#endif


void print_msgs(char* msg)
{
    printf("%s", msg);
}


std::optional<int>get_tb_result(Position pos, int max_pieces, EGDB_DRIVER* handle) {
    if (pos.hasJumps() || Bits::pop_count(pos.BP | pos.WP) > max_pieces)
        return std::nullopt;

   EGDB_NORMAL_BITBOARD board;
   board.white = pos.WP;
   board.black = pos.BP;
   board.king =pos.K;

   EGDB_BITBOARD normal;
   normal.normal = board;
   auto val = handle->lookup(handle, &normal, (pos.color == BLACK) ? EGDB_BLACK : EGDB_WHITE, 0);

   if (val == EGDB_UNKNOWN)
       return std::nullopt;

   if (val == EGDB_WIN)
       return pos.color;

   if (val == EGDB_LOSS)
       return -pos.color;


   return std::nullopt;
}

int get_game_result(std::vector<Position>& game) {
    MoveListe liste;
    getMoves(game.back(), liste);
    const size_t rep_count = std::count(game.begin(), game.end(), game.back());

    if (liste.length() == 0) {
        const int result = -game.back().color;
        return result;
    }
    else if (rep_count >= 3) {
        return 0;
    }
}

std::vector<Sample> get_rescored_game(std::vector<Position>& game, int max_pieces, EGDB_DRIVER* handle) {
    std::vector<Sample> sample_data;
   

        auto start_point = 0;
        while (start_point < game.size()) {
            std::optional<int>result;
            int k;
            for (k = start_point; k < game.size(); ++k) {
                result = get_tb_result(game[k], max_pieces, handle);
                if (result.has_value()) {
                    break;
                }
            }
            if (result.has_value()) {
                for (auto p = start_point; p <=k; ++p) {
                    Sample s;
                    s.position = game[p];
                    s.result = result.value();
                    sample_data.emplace_back(s);
                }
                start_point = k + 1;
            }
            else {
               const int result = get_game_result(game);
                for (auto p = start_point; p <game.size(); ++p) {
                    Sample s;
                    s.position = game[p];
                    s.result = result;
                    sample_data.emplace_back(s);
                }

                break;
            }

        }


        int res = sample_data.front().result;
        for (auto s : sample_data) {
            if (s.result != res) {

                for (auto p : sample_data) {
                    p.position.printPosition();
                    std::cout << "FenString: " << p.position.get_fen_string();
                    std::cout << "Result: " << p.result << std::endl;
                }
                for (auto i = 0; i < 3; ++i)
                    std::cout << "\n";

                break;
            }
        }

    return sample_data;
}


void create_samples_from_games(std::string games, std::string output, int max_pieces, EGDB_DRIVER* handle) {
    size_t uniq_count{ 0 };
    size_t total_count{ 0 };
    SampleFilter filter(5751035027, 10);
    std::vector<Sample> buffer;
    const size_t max_cap_buffer = 10000;

    size_t game_counter = 0;
    //first stepping through the games
    std::ifstream stream(games, std::ios::binary);
    std::ofstream out_stream(output, std::ios::binary);

    std::istream_iterator<Position> end;
    std::vector<Position> game;
    auto it = std::istream_iterator<Position>(stream);
    Position previous = *it;
    game.emplace_back(previous);
    ++it;
    for (; it != end; ++it) {
        Position pos = *it;
        const size_t piece_count = Bits::pop_count(pos.BP | pos.WP);
        const size_t prev_piec_count = Bits::pop_count(previous.BP | previous.WP);
        if (piece_count <= prev_piec_count) {
            game.emplace_back(pos);
        }
        else {
         
            auto samples = get_rescored_game(game, max_pieces, handle);
            std::copy(samples.begin(), samples.end(), std::back_inserter(buffer));
            game.clear();
            game_counter++;
        }
        previous = pos;
        if (buffer.size() >= max_cap_buffer) {
            std::copy(buffer.begin(), buffer.end(), std::ostream_iterator<Sample>(out_stream));
            buffer.clear();
        }
        total_count++;
    }
    std::cout << "Total Position: " << total_count << " after removing: " << uniq_count << std::endl;
    std::copy(buffer.begin(), buffer.end(), std::ostream_iterator<Sample>(out_stream));
}





int main(int argl, const char **argc) {

    int i, status, max_pieces, nerrors;
    EGDB_TYPE egdb_type;
    EGDB_DRIVER *handle;

    /* Check that db files are present, get db type and size. */
    status = egdb_identify(DB_PATH, &egdb_type, &max_pieces);

    if (status) {
        printf("No database found at %s\n", DB_PATH);
        return(1);
    }
    printf("Database type %d found with max pieces %d\n", egdb_type, max_pieces);

    /* Open database for probing. */
    handle = egdb_open(EGDB_NORMAL, max_pieces, 2000, DB_PATH, print_msgs);
    if (!handle) {
        printf("Error returned from egdb_open()\n");
        return(1);
    }

    std::string path("C:/Users/leagu/Downloads/test.games");
    std::string output("C:/Users/leagu/Downloads/testrescored.samples");
  
    create_samples_from_games(path, output, max_pieces, handle);


     handle->close(handle);

    return 0;
}
