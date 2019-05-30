


#include <vector>
#include <future>
#include <bitset>
#include <map>
#include "Transposition.h"
#include "GameLogic.h"
#include "BoardFactory.h"
#include "Perft.h"
#include <list>
#include <iterator>







int main(int argl, const char ** argc){

    std::cout<<"some small changes"<<std::endl;

    initialize();
    setHashSize(23);
    Board board;
    BoardFactory::setUpStartingPosition(board);

    board.printBoard();
    searchValue(board,MAX_PLY,200000,true);



}