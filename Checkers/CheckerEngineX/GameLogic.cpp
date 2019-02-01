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


#ifdef TRAIN
Weights<double> gameWeights;
#else

Weights<FixPoint<short, 4>> gameWeights;
#endif


MAKRO void initialize() {
    gameWeights.loadWeights("/home/robin/Checkers/Training/cmake-build-debug/Weights/popel.weights");
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
    getMoves(*board.getPosition(), easyMoves);
    if (easyMoves.length() == 1) {
        best = easyMoves.liste[0];
        board.makeMove(best);
        return EASY_MOVE;
    }
    TT.incrementAgeCounter();
    TT.clear();
    //Testing aging of TT entries

    timeOut = false;
    endTime = getSystemTime() + time;
    int i = 1;
    Value alpha = -INFINITE;
    Value beta = INFINITE;
    Value gameValue;

    while (i <= depth && i <= MAX_PLY) {

        Line currentPV;
        Value value = alphaBeta<PVNode>(board, alpha, beta, currentPV, 0, i, true);

        if (timeOut)
            break;



        mainPV = currentPV;
        best = mainPV[0];
        gameValue = value;

        if (print) {
            std::string temp = std::to_string(value.value) + "  ";
            temp += " Depth:" + std::to_string(i) + " | ";
            temp += " NodeCount: " + std::to_string(nodeCounter) + "\n";

            temp += mainPV.toString();
            temp += "\n";
            temp += "\n";
            std::cout << temp;
        }
        ++i;

        if (print) {

            std::cout << "Time needed: " << (getSystemTime() - endTime + time) << "\n";
        }
    }
    board.makeMove(best);

    return gameValue;
}

template<NodeType type>
Value quiescene(Board &board, Value alpha, Value beta, Line &pv, int ply) {
    constexpr bool inPVLine = (type == PVNode);
    assert(alpha.isEval() && beta.isEval());
    nodeCounter++;
    if (ply >= MAX_PLY) {
        return board.getMover() * gameWeights.evaluate(*board.getPosition());
    }

    MoveListe moves;
    getCaptures(*board.getPosition(), moves);
    Value bestValue = -INFINITE;

    if (moves.isEmpty()) {

        if (board.getPosition()->hasThreat()) {
            return alphaBeta<type>(board, alpha, beta, pv, ply, 1, false);
        }

        if (board.getPosition()->isWipe()) {
            return Value::loss(board.getMover(), ply);
        }

        bestValue = board.getMover() * gameWeights.evaluate(*board.getPosition());
        if (bestValue >= beta) {
            return bestValue;
        }
    }

    if (inPVLine && ply < mainPV.length()) {
        auto val = std::find(moves.begin(), moves.end(), mainPV[ply]);
        if (val != moves.end()) {
            std::swap(moves[0], *val);
        }
    }

    for (int i = 0; i < moves.length(); ++i) {
        Line localPV;
        board.makeMove(moves.liste[i]);
        Value value;
        if (i == 0) {
            value = ~quiescene<type>(board, ~beta, ~alpha, localPV, ply + 1);
        } else {

            value = ~quiescene<NONPV>(board, ~beta, ~alpha, localPV, ply + 1);
        }
        board.undoMove();

        if (value > bestValue) {
            bestValue = value;

            if (value >= beta) {
                break;
            }
            if (value > alpha) {
                pv.clear();
                pv.concat(moves[i], localPV);
                alpha = value;
            }

        }
    }

    return bestValue;
}


template<NodeType type>
Value
alphaBeta(Board &board, Value alpha, Value beta, Line &pv, int ply, int depth, bool prune) {
    constexpr bool inPVLine = (type == PVNode);
    const bool isRoot = (ply == 0);
    assert(alpha.isEval() && beta.isEval());
    if ((nodeCounter & 2047) == 0 && getSystemTime() >= endTime) {
        timeOut = true;
        return 0;
    }
    if (ply >= MAX_PLY) {
        return board.getMover() * gameWeights.evaluate(*board.getPosition());
    }

    if (ply > 0 && board.isRepetition()) {
        return 0;
    }

    if (depth <= 0) {
        return quiescene<type>(board, alpha, beta, pv, ply);
    }


    MoveListe sucessors;
    getMoves(*board.getPosition(), sucessors);
    if (sucessors.isEmpty()) {
        return Value::loss(board.getMover(), ply);
    }


#ifdef GENERATE
    //Randomly sorting the root moves
    if (ply == 0) {
    static std::mt19937 generator(getSystemTime());
    std::shuffle(sucessors.begin(),sucessors.end(),generator);
    }
#endif


    NodeInfo info;
#ifndef TRAIN
    TT.findHash(board.getCurrentKey(), depth, &alpha.value, &beta.value, info);
    info.value = info.value.valueFromTT(ply);
#endif


    if (!inPVLine && ply > 0 && alpha >= beta) {
        return info.value;
    }


    if (!inPVLine && prune && ply > 0 && depth >= 3 && beta <= INFINITE) {
        Value margin = 10 * depth;
        Value newBeta = addSafe(beta, margin);
        int newDepth = (depth * 40) / 100;
        Line local;
        Value value = alphaBeta<type>(board, newBeta - 1, newBeta, local, ply, newDepth, false);
        if (value >= newBeta) {
            return addSafe(value, ~margin);
        }
    }


    if (inPVLine && ply < mainPV.length()) {
        auto val = std::find(sucessors.begin(), sucessors.end(), mainPV[ply]);
        if (val != sucessors.end()) {
            std::swap(sucessors[0], *val);
        }
    }

    sucessors.sort(info.move, inPVLine, board.getMover());

    Value bestValue = -INFINITE;
    Move bestMove;
    Value alphaOrig = alpha;

    const int extension =0;

    int newDepth = depth - 1 + extension;

    for (int i = 0; i < sucessors.length(); ++i) {
        board.makeMove(sucessors[i]);
        Value value;
        Line localPV;
        if (i == 0) {
            value = ~alphaBeta<type>(board, ~beta, ~alpha, localPV, ply + 1, newDepth, prune);
        } else {
            int reduce = 0;
            if (depth >= 2 && i >= ((inPVLine) ? 3 : 1) && !sucessors[i].isPromotion() && !sucessors[i].isCapture()) {
                reduce = 1;
                if (i >= 4) {
                    reduce = 2;
                }
            }
            value = ~alphaBeta<NONPV>(board, ~alpha - 1, ~alpha, localPV, ply + 1, newDepth - reduce, prune);
            if (value > alpha && value < beta) {
                value = ~alphaBeta<NONPV>(board, ~beta, ~alpha, localPV, ply + 1, newDepth, prune);
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
                pv.concat(bestMove, localPV);
                alpha = value;
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
