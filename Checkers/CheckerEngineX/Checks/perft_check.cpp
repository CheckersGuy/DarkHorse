
#include <iostream>
#include "../Perft.h"
int main(){
    //Checking if the moveGenerator still works

    std::array<uint64_t,15> node_counts = {
            1u,7u,49u,302u,1469u,7361u,36768u,
            179740u,845931u,3963680u,18391564u,
            85242128u,388623673u,1766623630u,
            7978439499u
    };

    Zobrist::initializeZobrisKeys();
    Perft::table.setCapacity(1u<<26u);
    Board board;
    board=Position::getStartPosition();
    board.printBoard();
    std::cout<<std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    for(auto i=1;i<node_counts.size();++i){
        std::cout<<"Checking depth: "<<i<<" ";
        auto count = Perft::perftCheck(board.getPosition(),i);
        std::cout<<count<<" ";

        auto iter_time = std::chrono::high_resolution_clock::now();
        std::cout<<"Time: "<<(iter_time-start_time).count()/1000000<<std::endl;
        if(count != node_counts[i])
            return 1;
    }


    return 0;

}