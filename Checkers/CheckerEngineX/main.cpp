

#include <iostream>
#include "Perft.h"
#include "GameLogic.h"
#include "BoardFactory.h"




int main(int argLength, char **arguments) {

    initialize();
    setHashSize(25);
    Board test;
    BoardFactory::setUpStartingPosition(test);
    searchValue(test, 25, 200000, true);

/*
    for(int i=0;i<32;++i){
        Position pos;
        pos.BP=1<<i;
        pos.K=1<<i;
        pos.printPosition();
        std::cout<<"\n";
        pos.getColorFlip().printPosition();
        std::cout<<"#########################"<<"\n";
    }
*/


/*
    if(argLength > 1 && strcmp(arguments[1], "selfplay") == 0) {
        initialize();
        int hashSize = 0;
        std::cout << "Enter the hashSize to be used:" << "\n";
        std::cin >> hashSize;
        if (hashSize > 0) {
            TT.resize(hashSize);
        }
        Board test;
        int timeBlack, timeWhite, openingIndex;;


        std::cout << "Enter time for Black: " << "\n";
        std::cin >> timeBlack;

        std::cout << "Enter time for White: " << "\n";
        std::cin >> timeWhite;

        std::cout << "Play from the StartingPosition ?[y\\n]" << "\n";
        std::string tempString;
        std::cin >> tempString;
        if (tempString == "n") {
            std::cout << "choose opening" << "\n";
            std::cin >> openingIndex;
            BoardFactory::getOpeningPosition(openingIndex, test);
        } else if (tempString == "y") {
            BoardFactory::setUpStartingPosition(test);
        }

        test.printBoard();
        std::cout << "\n";


        for (int i = 0; i < 1000; ++i) {
            if (test.isRepetition()) {
                break;
            }
            MoveListe liste;
            getMoves(test,liste);
            if(liste.length()==0)
                break;

            if (test.getMover() == BLACK) {
                searchValue(test, MAX_PLY, timeBlack, true);

            } else {
                searchValue(test, MAX_PLY, timeWhite, true);
            }
#

            test.printBoard();
            std::cout << test.getBoardString();
            std::cout << "\n";
            std::cout << "\n";
        }

    }

    Board test;
    BoardFactory::setUpStartingPosition(test);

    setHashSize(24);

    test.printBoard();

    initialize();

    searchValue(test,MAX_PLY,2000000,true);

*/


    return 0;
}
