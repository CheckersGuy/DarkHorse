


#include <vector>
#include <future>
#include <bitset>
#include "Transposition.h"
#include "GameLogic.h"
#include "BoardFactory.h"
#include "Perft.h"


int main(int argLength, char **arguments) {


    initialize();
   /* using namespace Perft;
    table.setCapacity(1<<25);
    Board board;
    BoardFactory::setUpStartingPosition(board);
    ThreadPool pool(7);
    pool.setSplitDepth(18);

    pool.startThreads();

    pool.splitLoop(board,20,20);

    pool.waitAll();
    std::cout<<"PERFT: "<<pool.getNodeCounter()<<std::endl;
*/

    setHashSize(26);
    Board test;
    BoardFactory::setUpStartingPosition(test);

    searchValue(test,MAX_PLY,1500000,true);


    return 0;
}
