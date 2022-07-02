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
template<size_t size,typename Generator> constexpr auto get_lut1(Generator&& generator){
	using Type = decltype(generator(0));
	std::array<Type,size> result;

	for(auto i=0;i<size;++i){
		result[i]=generator(i);
	}
	return result;
};

constexpr std::array<size_t,32>To64 = get_lut1<32>([](size_t index){auto row = index/4; return ((row&1)==0) ? 2*index+1 : 2*index;});


int main(int argl, const char **argc) {


    initialize();
    Board board;
    use_classical(true);
    Statistics::mPicker.init();

 /* 
      network.load("form_network8.weights");
    network.addLayer(Layer{120, 256});
    network.addLayer(Layer{256, 32});
    network.addLayer(Layer{32, 32});
    network.addLayer(Layer{32, 1});
    network.init();   
  /* */ 
/*           TT.resize(23);
    board = Position::get_start_position();    
    //board = Position::pos_from_fen("W:WK5,29:BK3,K4,K16");
     
    board.get_position().make_move(11, 15);
    board.get_position().make_move(21, 17);
    board.get_position().make_move(9, 13);
    board.get_position().make_move(23, 19); 
    board.print_board(); 
   
    
      Move best;
    searchValue(board, best, MAX_PLY, 100000000, true);
    board.make_move(best);
    board.print_board();
    MoveListe liste;
    get_moves(board.get_position(), liste);   
          
          */

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
            board.make_move(move);
            std::cout << "update_ready" << "\n";
        } else if (current == "search") {
            std::string time_string;
            std::cin >> time_string;
            Move bestMove;

            MoveListe liste;
            get_moves(board.get_position(), liste);
            searchValue(board, bestMove, MAX_PLY, std::stoi(time_string), false);
            std::cout << "new_move" << "\n";
            std::cout << std::to_string(bestMove.get_from_index()) << "\n";
            std::cout << std::to_string(bestMove.get_to_index()) << "\n";
            uint32_t captures = bestMove.captures;
            while (captures) {
                std::cout << std::to_string(Bits::bitscan_foward(captures)) << "\n";
                captures &= captures - 1u;
            }
            std::cout << "end_move" << "\n";
            board.make_move(bestMove);
        } else if (current == "terminate") {
            //terminating the program
            break;
        }
    }


    return 0;
}
