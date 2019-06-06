


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







int main(int argl, const char **argc) {

    initialize();
    setHashSize(23);

    Board test;
    BoardFactory::setUpStartingPosition(test);

    searchValue(test,MAX_PLY,100000,true);






}