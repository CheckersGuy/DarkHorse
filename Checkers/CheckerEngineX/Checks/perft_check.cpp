
#include <iostream>
#include "../Perft.h"
int main(){
    //Checking if the moveGenerator still works

    Zobrist::initializeZobrisKeys();
    Perft::table.setCapacity(1u<<23);
    Board board;
    board=Position::getStartPosition();
    board.printBoard();
    std::cout<<std::endl;

    auto count = Perft::perftCheck(board,14);
    std::cout<<"Count: "<<count<<std::endl;

    if(count !=7978439499u){
        return 1;
    }
    return 0;

}