
#include <iostream>
#include "../Perft.h"

int main() {
    //Checking if the moveGenerator still works

    constexpr std::array<uint64_t, 29> node_counts = {
            1ull, 7ull, 49ull, 302ull, 1469ull, 7361ull, 36768ull,
            179740ull, 845931ull, 3963680ull, 18391564ull,
            85242128ull, 388623673ull, 1766623630ull,
            7978439499ull, 36263167175ull, 165629569428ull, 758818810990ull,
            3493881706141ull, 16114043592799ull, 74545030871553ull, 345100524480819ull,
            1602372721738102ull, 7437536860666213ull, 34651381875296000ull,
            161067479882075800ull, 752172458688067137ull, 3499844183628002605ull,
            16377718018836900735ull
    };

    Zobrist::init_zobrist_keys();
    Perft::table.set_capacity("32000mb");
    Board board;
    board = Position::get_start_position();
    board.print_board();
    std::cout << std::endl;


    PerftCallBack call_back;



    //std::cout<<"NumNodes: "<<call_back.num_nodes<<std::endl;





    for (auto i = 1; i < node_counts.size(); ++i) {
        std::cout << "Checking depth: " << i << " ";
        auto start_time = std::chrono::high_resolution_clock::now();
        call_back.num_nodes = 0;
        Perft::perft_check(board.get_position(),i,call_back);
        auto count = call_back.num_nodes;
        std::cout << count << " ";

        auto iter_time = std::chrono::high_resolution_clock::now();
        auto time = (iter_time - start_time).count() / 1000000;
        auto kilo_nodes = size_t{0};
        if(time>0){
            auto nodes_ms = call_back.num_nodes / time;
            kilo_nodes = nodes_ms;
        }
        std::cout << "Time: " << time << " KNodes: " << kilo_nodes << std::endl;
        if (count != node_counts[i])
            return 1;
    }


    return 0;

}