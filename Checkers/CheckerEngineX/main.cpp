



#include "Transposition.h"
#include "GameLogic.h"
#include "BoardFactory.h"

int main(int argLength, char **arguments) {


    for(int i=0;i<4;++i){
        Position test;

        test.BP=MASK_ROW<<4*i;
        test.WP=MASK_ROW<<4*(7-i);
        test.printPosition();
        std::cout<<std::endl;
    }



    /* initialize();
     using namespace Perft;


     for(int i=1;i<=21;++i){
         uint64_t startingTime = getSystemTime();
         std::cout <<"Depth: "<<i<<" "<< Perft::perftCount(i,1) << std::endl;
         std::cout<<"Time taken : "<<(getSystemTime()-startingTime)<<std::endl;
     }
 */


/*
    std::vector<bool> test;

    uint16_t encoding = 16256 + 4 + 8;

    while (encoding) {
        test.emplace_back(((encoding % 2) == 1));
        encoding /= 2;
    }

    for (int i = test.size() - 1; i >= 0; i--) {
        if (test[i])
            std::cout << 1;
        else
            std::cout << 0;
    }
*/

   /* Entry another;

    for(int i=0;i<4;++i){
        another.setAgeCounter(i);

        std::cout<<another.getAgeCounter()<<std::endl;
        std::cout<<another.getFlag()<<std::endl;
        std::cout<<another.getDepth()<<std::endl;
        std::cout<<std::endl;
        std::cout<<std::endl;

    }*/






   getchar();
    return 0;
}
