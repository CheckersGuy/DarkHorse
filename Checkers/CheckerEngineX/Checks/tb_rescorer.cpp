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
#include "../../Training/BloomFilter.h"
#include "../../Training/Util/Compress.h"
#include "CmdParser.h"

#define DB_PATH "D:\kr_english_wld"



void print_msgs(char *msg) {
    printf("%s", msg);
}


Result get_tb_result(Position pos, int max_pieces, EGDB_DRIVER *handle) {
    if (pos.has_jumps() || Bits::pop_count(pos.BP | pos.WP)>max_pieces)
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

Result get_game_result(Game& game) {
    Position end = game.get_last_position();
/*     end.print_position();
    std::cout<<std::endl; */
    MoveListe liste;
    get_moves(end, liste);
    const size_t rep_count = std::count(game.begin(), game.end(),end);
    if (liste.length() == 0) {
        return (end.color == BLACK) ? WHITE_WON : BLACK_WON;
    }
    else if (rep_count >= 3) {
        return DRAW;
    }
    return UNKNOWN;
}

Result get_game_result(std::vector<Position> &game) {
    MoveListe liste;
    get_moves(game.back(), liste);
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
        if (!previous.has_jumps(previous.get_color())) {

            Move move;
            move.from = (previous.BP & (~pos.BP)) | (previous.WP & (~pos.WP));
            move.to = (pos.BP & (~previous.BP)) | (pos.WP & (~previous.WP));
            Position copy;
            copy = previous;
            copy.make_move(move);
            if (copy != pos) {
                return std::vector<Sample>{};
            }


            sample_data[k - 1].move = Statistics::mPicker.get_move_encoding(sample_data[k - 1].position.get_color(),
                                                                            move);
            if (sample_data[k - 1].move >= 100) {
                std::cerr << "Error move: " << sample_data[k - 1].move << std::endl;
                std::exit(-1);
            }
        }


    }

    return sample_data;

}

Game get_rescored_game(Game& game, int max_pieces, EGDB_DRIVER* handle) {
    const Result result = get_game_result(game);
    game.result = result;
    //result may still be unknown

    for (auto i = 0; i < game.indices.size(); ++i) {
        Game::encode_result(game.indices[i], game.result);
    }
    //there may be an easier way to do this
    std::vector<Position>positions;
    for (auto p : game) {
        positions.emplace_back(p);
    }

    auto rescored_samples = get_rescored_game(positions, max_pieces, handle);

    for (auto i = 0; i <game.indices.size(); ++i) {
        Sample s = rescored_samples[i];
        Game::encode_result(game.indices[i], s.result);
    }
    Sample last = rescored_samples[game.indices.size()];
    game.result = last.result;

    std::vector<Sample> check_samples;
    game.extract_samples_test(std::back_inserter(check_samples));

    for (auto i = 0; i < check_samples.size(); ++i) {
        Sample s = rescored_samples[i];
       // std::cout << "Result: " << s.result << std::endl;
        //s.position.print_position();
        //std::cout << "Result: " << s.result << std::endl;
       // s.position.print_position();
        if (check_samples[i].position != rescored_samples[i].position) {
            std::cerr << "Error in rescoring the game" << std::endl;
            std::exit(-1);
        }
        if (check_samples[i].result != rescored_samples[i].result) {
            std::cerr << "Error in rescoring the game" << std::endl;
            std::exit(-1);
        }
    }


    return game;

}


void create_samples_from_games(std::string games, std::string output, int max_pieces, EGDB_DRIVER *handle) {
  
    std::ifstream stream(games,std::ios::binary);
    if(!stream.good()){
        std::cerr<<"Could not open input stream"<<std::endl;
        std::exit(-1);
    }
    std::ofstream out_stream(output, std::ios::binary);
    if(!out_stream.good()){
        std::cerr<<"Could not open output stream"<<std::endl;
        std::exit(-1);
    }
    std::istream_iterator<Game> begin(stream);
    std::istream_iterator<Game>end;
    size_t counter{ 0 };
    std::for_each(begin, end, [&](Game game) {
        if(game.result!=UNKNOWN){
            out_stream << game;
            return;
        }
            
        auto r = get_rescored_game(game, max_pieces, handle);
        out_stream << r;
        });

}


int main(int argl, const char **argc) {


    CmdParser parser(argl,argc);
    parser.parse_command_line();
    std::string input,output;

    if(parser.has_option("input") && parser.has_option("output")){
        input = parser.as<std::string>("input");
        output = parser.as<std::string>("output");
    }else{
        std::exit(-1);
    }

    int i, status, max_pieces, nerrors;
    EGDB_TYPE egdb_type;
    EGDB_DRIVER *handle;

    /* Check that db files are present, get db type and size. */
    status = egdb_identify(DB_PATH, &egdb_type, &max_pieces);
    std::cout<<"MAX_PIECES: "<<max_pieces<<std::endl;

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
    std::cout<<"Starting Rescoring the training data"<<std::endl;
  

    create_samples_from_games(input, output, max_pieces, handle);

    handle->close(handle);

    return 0;
}
