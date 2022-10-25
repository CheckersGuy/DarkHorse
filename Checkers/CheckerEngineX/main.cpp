#include <vector>
#include "Transposition.h"
#include "GameLogic.h"
#include "Perft.h"

inline Position posFromString(const std::string &pos) {
    Position result;
    for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t current = 1u << i;
        if (pos[i] == '1') {
            result.BP |= current;
        } else if (pos[i] == '2') {
            result.WP |= current;
        } else if (pos[i] == '3') {
            result.K |= current;
            result.BP |= current;
        } else if (pos[i] == '4') {
            result.K |= current;
            result.WP |= current;
        }
    }
    if (pos[32] == 'B') {
        result.color = BLACK;
    } else {
        result.color = WHITE;
    }
    result.key = Zobrist::generate_key(result);
    return result;
}


//game-generation

#include <Network.h>
#include <types.h>
#include "CmdParser.h"

void test_policy_output(Position pos,bool show_all){
	pos.print_position();
	auto& net = Statistics::mPicker.policy;
	auto result =Statistics::mPicker.policy.compute_incre_forward_pass(pos);
	MoveListe liste;
	get_moves(pos,liste);
	for(auto m : liste){
			auto index = Statistics::mPicker.get_move_encoding(pos.get_color(),m);
			std::cout<<"From: "<<m.get_from_index()<<" To: "<<m.get_to_index()<< "Score: "<<net[index];
			std::cout<<std::endl;
	}

	if(show_all){
			for(auto i=0;i<128;++i){
					std::cout<<"Index: "<<i<<" Score: "<<net[i]<<std::endl;
			}
	}

}




int main(int argl, const char **argc) {
    
	CmdParser parser(argl,argc);
    parser.parse_command_line();
    Board board;
    initialize();
	Statistics::mPicker.init();
//	test_policy_output(Position::get_start_position(),true);
//	return 0;

    int time,depth,hash_size;
    std::string net_file;

    if(parser.has_option("network")) {
        net_file= parser.as<std::string>("network");
    } else {
        net_file ="bigagain5.quant";
    }

    network.addLayer(Layer{120, 1024});
    network.addLayer(Layer{1024, 8});
    network.addLayer(Layer{8, 32});
    network.addLayer(Layer{32, 1});
    network.load(net_file);
    network.init();
    if (parser.has_option("search"))

    {
        if (parser.has_option("time"))
        {
            time = parser.as<int>("time");
        }
        else
        {
            time = 100000000;
        }

        if (parser.has_option("depth"))
        {
            depth = parser.as<int>("depth");
        }
        else
        {
            depth = MAX_PLY;
        }
        if (parser.has_option("hash_size"))
        {
            hash_size = parser.as<int>("hash_size");
        }
        else
        {
            hash_size = 21;
        }

        if (parser.has_option("position"))
        {
            auto pos_string = parser.as<std::string>("position");
            board.get_position() = Position::pos_from_fen(pos_string);
        }
        else
        {
            board.get_position() = Position::get_start_position();
        }



        TT.resize(hash_size);


        Move best;
        searchValue(board, best, depth, time, true, std::cout);
        board.play_move(best);
        board.print_board();
        MoveListe liste;
        get_moves(board.get_position(), liste);
        return 0;
    }



    std::string current;
    while (std::cin >> current) {
        if (current == "init") {
            TT.age_counter = 0u;
            initialize();
            std::string hash_string;
            std::cin >> hash_string;
            const int hash_size = std::stoi(hash_string);
            TT.resize(hash_size);
            std::cout << "init_ready" << "\n";
        } else if (current == "new_game") {
            TT.clear();
            TT.age_counter = 0u;
            network.init();
            board = Board{};
            std::string position;
            std::cin >> position;
            Position pos = posFromString(position);
            board = pos;
            std::cout << "game_ready" << "\n";
        } else if (current == "new_move") {
            //opponent made a move and we need to update the board
            Move move;
            std::vector <uint32_t> squares;
            std::string line;
            std::cin >> line;
            while (!line.empty()) {
                if (line == "end_move")
                    break;
                squares.emplace_back(std::stoi(line));
                std::cin >> line;
            }
            move.from = 1u << squares[0];
            move.to = 1u << squares[1];
            for (auto i = 2; i < squares.size(); ++i) {
                move.captures |= 1u << squares[i];
            }
            board.play_move(move);
            std::cout << "update_ready" << "\n";
        } else if (current == "search") {
            std::string time_string;
            std::cin >> time_string;
            Move bestMove;
            searchValue(board, bestMove, MAX_PLY, std::stoi(time_string), false,std::cout);
            std::cout << "new_move" << "\n";
            std::cout << std::to_string(bestMove.get_from_index()) << "\n";
            std::cout << std::to_string(bestMove.get_to_index()) << "\n";
            uint32_t captures = bestMove.captures;
            while (captures) {
                std::cout << std::to_string(Bits::bitscan_foward(captures)) << "\n";
                captures &= captures - 1u;
            }
            std::cout << "end_move" << "\n";
            board.play_move(bestMove);
        } else if (current == "terminate") {
            //terminating the program
            break;
        }
    }
}
