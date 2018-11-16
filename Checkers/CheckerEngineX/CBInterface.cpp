//
// Created by Robin on 02.02.2018.
//

#include "CBInterface.h"

Board playBoard;
int convIndex[32] = {0, 2, 4, 6, 9, 11, 13, 15, 16, 18, 20, 22, 25, 27, 29, 31, 32, 34, 36, 38, 41, 43, 45, 47, 48, 50,
                     52, 54, 57, 59, 61, 63};
char *output = nullptr;
bool initialized=false;

int getmove(int board[8][8], int color, double maxtime, char str[1024], int *playnow, int info, int unused,
            struct CBmove *move) {
    output = str;
    Position *pos = &playBoard.pStack[playBoard.pCounter];
    pos->BP=0;
    pos->WP=0;
    pos->K=0;
    if (isBitSet(info, 0)) {
        sprintf(str, "interupt");
        //Natural course of the game has been interrupted;
        playBoard.pCounter = 0;
    }


    if (color == CBWHITE) {
        playBoard.pStack[playBoard.pCounter].color = WHITE;
    } else {
        playBoard.pStack[playBoard.pCounter].color = BLACK;
    }
    sprintf(str, "Some testing");
    if (*playnow) {
        playBoard.pCounter=0;

    }
    pos->BP=0;
    pos->WP=0;
    pos->K=0;
    if (!initialized) {
        sprintf(str, "initializing the engine");
        initialized=true;
        initialize();
    }
    //Trying to setup the startingPosition
    int counter = 0;
    std::string testString;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if ((j + i) % 2 == 0) {
                uint32_t mask = 1 << S[counter];
                if (board[j][i] == CBMAN + CBWHITE) {
                    pos->WP |= mask;
                } else if (board[j][i] == CBMAN + CBBLACK) {
                    pos->BP |= mask;
                } else if (board[j][i] == CBKING + CBBLACK) {
                    pos->BP |= mask;
                    pos->K |= mask;
                } else if (board[j][i] == CBKING + CBWHITE) {
                    pos->WP |= mask;
                    pos->K |= mask;
                }
                counter++;
            }
        }
    }

    MoveListe liste;
    getMoves(*playBoard.getPosition(), liste);
    Move bestMove;
    Value value = searchValue(playBoard, bestMove, 128, static_cast<uint32_t>(maxtime * 1000), false);

    int from = bestMove.getFrom();
    from = convIndex[S[from]];
    int to = bestMove.getTo();
    to = convIndex[S[to]];
    board[from % 8][from / 8] = CBFREE;

    if (color == CBBLACK) {

        if (bestMove.isPromotion() || bestMove.getPieceType() == 1) {
            board[to % 8][to / 8] = CBBLACK + CBKING;
        } else {
            board[to % 8][to / 8] = CBBLACK + CBMAN;
        }
    } else {
        if (bestMove.isPromotion() || bestMove.getPieceType() == 1) {
            board[to % 8][to / 8] = CBWHITE + CBKING;
        } else {
            board[to % 8][to / 8] = CBWHITE + CBMAN;
        }
    }
    if (bestMove.isCapture()) {
        uint32_t captures = bestMove.captures;
        while (captures) {
            uint32_t index = convIndex[S[__tzcnt_u32(captures)]];
            captures &= captures - 1;
            board[index % 8][index / 8] = CBFREE;
        }
    }
    testString += std::to_string(to % 8) + "|" + std::to_string(from / 8);
    sprintf(str, ((std::to_string(bestMove.getFrom() + 1) + "|" + std::to_string(bestMove.getTo() + 1) + "TEST: " +
                   std::to_string(from % 8) + "|" + std::to_string(from / 8) + " To: " + std::to_string(to % 8) + "|" +
                   std::to_string(to / 8)).c_str()));

    playBoard.pCounter++;

    return 3;
};





int enginecommand(char str[256], char reply[1024]) {
    const uint32_t mbSize =10000000;
    if(strcmp(str,"about")==0){
        sprintf(reply,"This is a test version");
        return 1;
    }
    if(strcmp(str,"get hashsize")==0){
        //calculate the size of the hashtable in MB
        uint32_t  hashSize =(TT.getCapacity())*2*sizeof(Entry);
        hashSize=hashSize/mbSize;
        int power =0;
        while((hashSize/2)){
            hashSize=hashSize/2;
            power++;
        }
        sprintf(reply,std::to_string(1<<power).c_str());
        return 1;
    }

    if(strcmp(str,"name")==0){
        sprintf(reply,"CheckerEngineX");
        return 1;
    }
    if(strcmp(str,"set hashsize 2048")==0) {
        TT.resize(((1 << 9) * mbSize) / (sizeof(Entry) * 2));
        return 1;
    }


    return 0;
}