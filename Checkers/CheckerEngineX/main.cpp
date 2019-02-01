


#include <vector>
#include <future>
#include "Transposition.h"
#include "GameLogic.h"
#include "BoardFactory.h"
#include "Perft.h"


int main(int argLength, char **arguments) {


  /*  FixPoint<short,4> test;
    test=600;
*/


    /*
    int phase =8;
    FixPoint<short,4>opening =2;
    FixPoint<short,4>ending =0;

    FixPoint<short,4>opFactor=(phase);
    opFactor/=24;

    FixPoint<short,4>endFactor=(24-phase);
    endFactor/=24;

    FixPoint<short,4>eval =opFactor*opening+ending*endFactor;

    std::cout<<eval<<std::endl;
*/
    initialize();
    using namespace Perft;
    table.setCapacity(1<<25);
    Board board;
    BoardFactory::setUpStartingPosition(board);
    ThreadPool pool(7);
    pool.setSplitDepth(18);

    pool.startThreads();

    pool.splitLoop(board,20,20);

    pool.waitAll();
    std::cout<<"PERFT: "<<pool.getNodeCounter()<<std::endl;



    return 0;
}
