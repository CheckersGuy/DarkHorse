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


void print_msgs(char *msg) {
    printf("%s", msg);
}


Result get_tb_result(Position pos, int max_pieces, EGDB_DRIVER *handle) {
    if (pos.hasJumps() || Bits::pop_count(pos.BP | pos.WP) > max_pieces)
        return UNKNOWN;

    EGDB_NORMAL_BITBOARD board;
    board.white = pos.WP;
    board.black = pos.BP;
    board.king = pos.K;

    EGDB_BITBOARD normal;
    normal.normal = board;
    auto val = handle->lookup(handle, &normal, ((pos.color == BLACK) ? EGDB_BLACK : EGDB_WHITE), 0);

    if (val == EGDB_UNKNOWN)
        return UNKNOWN;

    if (val == EGDB_WIN)
        return (pos.color == BLACK) ? BLACK_WON : WHITE_WON;

    if (val == EGDB_LOSS)
        return (pos.color == BLACK) ? WHITE_WON : BLACK_WON;

    if (val == EGDB_DRAW)
        return DRAW;


    return UNKNOWN;
}

Result get_game_result(std::vector<Position> &game) {
    MoveListe liste;
    getMoves(game.back(), liste);
    const size_t rep_count = std::count(game.begin(), game.end(), game.back());

    if (liste.length() == 0) {
        return (game.back().color == BLACK) ? WHITE_WON : BLACK_WON;
    } else if (rep_count >= 3) {
        return DRAW;
    }
    return UNKNOWN;
}

std::vector<Sample> get_rescored_game(std::vector<Position> &game, int max_pieces, EGDB_DRIVER *handle) {
    std::vector<Sample> sample_data;
    const Result result = get_game_result(game);

    for (auto p: game) {
        Sample s;
        s.position = p;
        sample_data.emplace_back(s);
    }
    int last_stop = -1;
    for (auto i = 0; i < game.size(); ++i) {
        auto tb_result = get_tb_result(game[i], max_pieces, handle);
        if (tb_result == UNKNOWN)
            continue;
        for (int k = i; k > last_stop; k--) {
            Sample s;
            s.position = game[k];
            s.result = tb_result;
            sample_data[k] = s;
        }
        last_stop = i;
    }
    for (auto s : sample_data) {
        if(s.result !=UNKNOWN)
            continue;
        s.result = result;
    }

    //Getting the moves played
    for (auto k = 1; k < sample_data.size(); ++k) {
        Position pos = sample_data[k].position;
        Position previous = sample_data[k - 1].position;
        if (!previous.hasJumps(previous.getColor())) {

            Move move;
            move.from = (previous.BP & (~pos.BP)) | (previous.WP & (~pos.WP));
            move.to = (pos.BP & (~previous.BP)) | (pos.WP & (~previous.WP));
            Position copy;
            copy = previous;
            copy.makeMove(move);
            if (copy != pos) {
                return std::vector<Sample>{};
            }


            sample_data[k - 1].move = Statistics::mPicker.get_move_encoding(sample_data[k - 1].position.getColor(),
                                                                            move);
            if (sample_data[k - 1].move >= 100) {
                std::cerr << "Error move: " << sample_data[k - 1].move << std::endl;
                std::exit(-1);
            }
        }


    }

    return sample_data;

}


void create_samples_from_games(std::string games, std::string output, int max_pieces, EGDB_DRIVER *handle) {
    size_t uniq_count{0};
    size_t total_count{0};
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
        MoveListe liste;
        getMoves(previous, liste);

        if (piece_count <= prev_piec_count && liste.length() > 0) {
            game.emplace_back(pos);
        } else {

            auto samples = get_rescored_game(game, max_pieces, handle);

            for (auto s: samples) {
                total_count++;
                buffer.emplace_back(s);
            }
            game.clear();
            game_counter++;
        }
        previous = pos;
        if (buffer.size() >= max_cap_buffer) {
            std::cout << "Clearing buffer" << std::endl;
            std::copy(buffer.begin(), buffer.end(), std::ostream_iterator<Sample>(out_stream));
            buffer.clear();
        }

    }
    std::cout << "Total Position: " << std::endl;
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
        return (1);
    }
    printf("Database type %d found with max pieces %d\n", egdb_type, max_pieces);

    /* Open database for probing. */
    handle = egdb_open(EGDB_NORMAL, max_pieces, 2000, DB_PATH, print_msgs);
    if (!handle) {
        printf("Error returned from egdb_open()\n");
        return (1);
    }
    std::string in_file("C:\\Users\\leagu\\Downloads\\small_dataset.games");
    std::string out_file("C:\\Users\\leagu\\Downloads\\small_dataset.train");


    create_samples_from_games(in_file, out_file, max_pieces, handle);


    handle->close(handle);

    return 0;
}
