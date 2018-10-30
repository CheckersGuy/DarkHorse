//
// Created by Robin on 11.05.2017.
//

#include "BoardFactory.h"

namespace BoardFactory {


    void setUpPosition(Board &board, Position pos) {
        board.getPosition()->BP = pos.BP;
        board.getPosition()->WP = pos.WP;
        board.getPosition()->K = pos.K;
        board.getPosition()->color = pos.color;
        board.getPosition()->key = Zobrist::generateKey(*board.getPosition(), pos.color);

    }

    void setUpStartingPosition(Board &current) {
        current.pCounter = 0;
        Position *pos = current.getPosition();
        pos->BP = 0;
        pos->WP = 0;
        pos->K = 0;
        for (int i = 0; i <= 11; i++) {
            pos->BP |= 1 << S[i];
        }
        for (int i = 20; i <= 31; i++) {
            pos->WP |= 1 << S[i];
        }
        current.pStack[current.pCounter].color = BLACK;
        pos->key = Zobrist::generateKey(*current.getPosition(), current.getMover());
    }


    void getOpeningPosition(int index, Board &board) {
        board.pCounter = 0;
        Position *pos = board.getPosition();
        std::ifstream myStream;
        myStream.open("openingFile");
        if (!myStream.good()) {
            std::cout << "Error could not find the opening file" << std::endl;
            return;
        }
        int currentIndex = 0;
        while (!myStream.eof()) {
            std::string currentLine;
            std::getline(myStream, currentLine);
            if (currentIndex == index) {
                BoardFactory::getBoardFromString(board, currentLine);
                break;
            }
            currentIndex++;
        }
        pos->key = Zobrist::generateKey(*board.getPosition(), board.getMover());
        myStream.close();
    }

    void getBoardFromString(Board &board, std::string string) {
        board.pCounter = 0;
        Position *pos = board.getPosition();
        const std::string whitePawn = "WP";
        const std::string blackPawn = "BP";
        const std::string whiteKing = "WK";
        const std::string blackKing = "BK";
        const std::string empty = "EM";
        for (int i = 0; i <= 31; i++) {
            const int maske = 1 << i;
            std::string current = string.substr(2 * i, 2);
            if (current == whitePawn) {
                pos->WP |= maske;
            }
            if (current == whiteKing) {
                pos->WP |= maske;
                pos->K |= maske;
            }
            if (current == blackPawn) {
                pos->BP |= maske;
            }
            if (current == blackKing) {
                pos->K |= maske;
                pos->BP |= maske;
            }
        }
        //Color

        std::string color = string.substr(2 * 32, 2);
        if (color == "WM") {
            board.pStack[board.pCounter].color = WHITE;
        } else {
            board.pStack[board.pCounter].color = BLACK;
        }
        pos->key = Zobrist::generateKey(*board.getPosition(), board.getMover());
    }


}

