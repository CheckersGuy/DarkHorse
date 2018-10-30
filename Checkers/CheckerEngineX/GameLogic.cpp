//
// Created by Robin on 10.06.2017.
//

#include "GameLogic.h"


uint64_t nodeCounter = 0;
Line mainPV;



extern char *output;
bool timeOut = false;
uint64_t endTime = 1000000000;

MAKRO void setHashSize(uint32_t hash) {
    TT.resize(hash);
}

GameWeights gameWeights;

MAKRO void initialize() {
    gameWeights.loadWeights("Weights/test.weights");
    TT.resize(21);
    Zobrist::initializeZobrisKeys();
}


Value searchValue(Board &board, int depth, uint32_t time, bool print) {
    Move best;
    return searchValue(board, best, depth, time, print);
}

MAKRO Value searchValue(Board &board, Move &best, int depth, uint32_t time, bool print) {

    Statistics::mPicker.clearScores();
    nodeCounter = 0;
    MoveListe easyMoves;
    getMoves(board, easyMoves);
    Value gameValue;

    Value alpha = -INFINITE;
    Value beta = INFINITE;

    if (easyMoves.length() == 1) {
        best = easyMoves.liste[0];
        board.makeMove(best);
        return EASY_MOVE;
    }
    TT.clear();
    timeOut = false;
    endTime = getSystemTime() + time;
    for (int i = 1; i <= depth && i <= MAX_PLY; ++i) {
        Line currentPV;
        Value value = alphaBeta(board, alpha, beta, currentPV, true, 0, i, true);

        mainPV = currentPV;
        if (timeOut)
            break;

        best =mainPV[0];
        gameValue = value;
        uint64_t currentValue = getSystemTime();
        if (print) {
            std::string temp = std::to_string(gameValue.value) + "  ";
            temp += " Depth:" + std::to_string(i) + " ";
            temp += " Time: " + std::to_string(((((time + currentValue - endTime)) / 1000.0))) + "s  ";

            temp += mainPV.toString();
            temp += "\n";
            temp += "\n";
            std::cout << temp;
        }
    }
    if (print) {

        std::cout << "Time needed: " << (getSystemTime() - endTime + time) << "\n";
    }
    board.makeMove(best);

    return gameValue;
}


Value quiescene(Board &board, Value alpha, Value beta, Line &pv, int ply) {
    assert(alpha.isEval() && beta.isEval());
    nodeCounter++;
    if (ply >= MAX_PLY) {
        return board.getMover() * gameWeights.evaluate(*board.getPosition());
    }

    MoveListe moves;
    getCaptures(board, moves);
    Value bestValue = -INFINITE;

    if (moves.length() == 0) {

        if (board.getPosition()->hasThreat()) {
            return alphaBeta(board, alpha, beta, pv, alpha != beta - 1, ply, 1, false);
        }

        if (board.getPosition()->isWipe()) {
            return Value::loss(board.getMover(), ply);
        }

        bestValue = board.getMover() * gameWeights.evaluate(*board.getPosition());
        if (bestValue >= beta) {
            return bestValue;
        }
    }


    for (int i = 0; i < moves.length(); ++i) {
        Line localPV;
        board.makeMove(moves.liste[i]);
        Value value = ~quiescene(board, ~beta, ~alpha, localPV, ply + 1);
        board.undoMove();

        if (value > bestValue) {
            bestValue = value;
            if (value >= beta) {
                break;
            }
            if (value > alpha) {
                alpha = value;
                pv.concat(moves[i],localPV);
            }

        }
    }
    return bestValue;
}


Value
alphaBeta(Board &board, Value alpha, Value beta, Line &pv, bool inPVLine, int ply, int depth, bool prune) {

    assert(alpha.isEval() && beta.isEval());
    if ((nodeCounter & 4095) == 0 && getSystemTime() >= endTime) {
        timeOut = true;
        return 0;
    }
    if (ply >= MAX_PLY) {
        return board.getMover() * gameWeights.evaluate(*board.getPosition());
    }


    if (depth <= 0) {
        return quiescene(board, alpha, beta, pv, ply);
    }

    if (ply > 0 && board.isRepetition()) {
        return -board.getMover();
    }


    MoveListe sucessors;
    getMoves(board, sucessors);
    if (sucessors.length() == 0) {
        return Value::loss(board.getMover(), ply);
    }

#ifdef GENERATE
    //Randomly sorting the root moves
    if (ply == 0) {
    std::mt19937 generator(getSystemTime());
    std::shuffle(sucessors.begin(),sucessors.end(),generator);
#endif


    NodeInfo info;

    TT.findHash(board.getCurrentKey(), depth, &alpha.value, &beta.value, info);
    info.value = info.value.valueFromTT(ply);

    if (!inPVLine && ply > 0 && alpha >= beta) {
        return info.value;
    }


    if (prune && !inPVLine && ply > 0 && depth >= 5) {
        Value margin = std::min(10 * depth, 250);
        Value newBeta = addSafe(beta, margin);
        int newDepth = (depth * 40) / 100;;
        Value value = alphaBeta(board, newBeta - 1, newBeta, pv, false, ply, newDepth, false);
        if (value >= newBeta) {
            return addSafe(value, ~margin);
        }
    }


    if (inPVLine && ply < mainPV.length()) {
        int index = sucessors.findIndex(mainPV[ply]);
        if (index > 0) {
            sucessors.swap(0, index);
        }
    }
    sucessors.sort(info.move, inPVLine, board.getMover());

    Value bestValue = -INFINITE;
    Move bestMove;
    Value alphaOrig = alpha;

    int newDepth = depth - 1;


    for (int i = 0; i < sucessors.length(); ++i) {
        board.makeMove(sucessors[i]);
        Value value;
        Line localPV;
        if (i == 0) {
            value = ~alphaBeta(board, ~beta, ~alpha, localPV, inPVLine, ply + 1, newDepth, prune);
        } else {
            int reduce = 0;
            if (depth >= 2 && i > ((inPVLine) ? 1 : 0) && !sucessors[i].isPromotion() && !sucessors[i].isCapture()) {
                reduce = 1;
                if (i > 4) {
                    reduce = 2;
                }
            }
            value = ~alphaBeta(board, ~alpha - 1, ~alpha, localPV, false, ply + 1, newDepth - reduce, prune);
            if (value > alpha && value < beta) {
                value = ~alphaBeta(board, ~beta, ~alpha, localPV, false, ply + 1, newDepth, prune);
            }
        }
        board.undoMove();
        if (value > bestValue) {
            bestValue = value;
            bestMove = sucessors.liste[i];
            if (value >= beta) {
                Statistics::mPicker.updateHHScore(sucessors.liste[i], board.getMover(), depth);
                Statistics::mPicker.updateBFScore(sucessors.liste, i, board.getMover(), depth);
                break;
            }
            if (value > alpha) {
                pv.clear();
                alpha = value;
                pv.concat(bestMove, localPV);
            }
        }
    }
    if (bestValue <= alphaOrig) {
        TT.storeHash(bestValue.toTT(ply), board.getCurrentKey(), TT_UPPER, depth, bestMove);
    } else if (bestValue >= beta) {
        TT.storeHash(bestValue.toTT(ply), board.getCurrentKey(), TT_LOWER, depth, bestMove);
    } else {
        TT.storeHash(bestValue.toTT(ply), board.getCurrentKey(), TT_EXACT, depth, bestMove);
    }
    return bestValue;
}
