
#include <iostream>
#include "../Perft.h"
int main(){
    //Checking if the moveGenerator still works

    const int depth=14;

    Zobrist::initializeZobrisKeys();
    Perft::table.setCapacity(1u<<27u);
    Board board;
    board=Position::getStartPosition();
    board.printBoard();
    std::cout<<std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    auto count = Perft::perftCheck(board.getPosition(),depth);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto diff = t2-t1;
    std::cout<<"Depth: "<<depth<<std::endl;
    std::cout<<"Count: "<<count<<std::endl;
    std::cout<<"Time: "<<(diff.count())/1000000<<std::endl;
    if(count !=7978439499u){
        return 1;
    }
    return 0;

}