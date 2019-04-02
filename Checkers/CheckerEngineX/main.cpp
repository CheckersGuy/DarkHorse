


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


int main(int argLength, char **arguments) {
    std::cout<<"Main Branch"<<std::endl;

    initialize();
    setHashSize(22);
    Board board;
    BoardFactory::setUpStartingPosition(board);

    board.printBoard();
    std::cout<<std::endl;
    searchValue(board,MAX_PLY,1000000,true);

    return 0;
}
