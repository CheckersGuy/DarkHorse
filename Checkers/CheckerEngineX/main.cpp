


#include <vector>
#include <future>
#include <bitset>
#include <map>
#include "Transposition.h"
#include "GameLogic.h"
#include "BoardFactory.h"
#include <list>
#include <iterator>
#include <algorithm>
#include "Perft.h"






int main(int argl, const char **argc) {

    initialize();
    /*setHashSize(23  );
    Board test;
    BoardFactory::setUpStartingPosition(test);
    searchValue(test,MAX_PLY,100000,true);
*/



    Board board;
    BoardFactory::setUpStartingPosition(board);

    Perft::table.setCapacity(1<<26);

    auto t1=std::chrono::high_resolution_clock::now();
    uint64_t count = Perft::perftCheck(board,15);
    auto t2=std::chrono::high_resolution_clock::now();
    auto c=t2-t1;
    std::cout<<count<<std::endl;
    std::cout<<"Time passed: "<<c.count()/1000000<<std::endl;


}