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
Weights<int> gameWeights;
#endif


MAKRO void initialize() {
#ifdef __EMSCRIPTEN__
    Bits::set_up_bitscan();
#endif
    gameWeights.loadWeights<uint32_t>("/home/robin/DarkHorse/Training/cmake-build-debug/failSave.weights");
    std::cerr << "loaded weights" << std::endl;
    Zobrist::initializeZobrisKeys();
}


Value searchValue(Board &board, int depth, uint32_t time, bool print) {
    Move best;
    return searchValue(board, best, depth, time, print);
}


MAKRO Value searchValue(Board &board, Move &best, int depth, uint32_t time, bool print) {
    Statistics::mPicker.clearScores();
    nodeCounter = 0;
    mainPV.clear();
    TT.clear();

    MoveListe easyMoves;
    getMoves(board.getPosition(), easyMoves);
   /* if (easyMoves.length() == 1) {
        best = easyMoves[0];
        return EASY_MOVE;
    }*/

    timeOut = false;
    endTime = getSystemTime() + time;
    int i = 1;

    Value alpha = -INFINITE;
    Value beta = INFINITE;
    Value value;
    Value eval;

    while (i <= depth && i <= MAX_PLY) {
        Line currentPV;
        value = alphaBeta<PVNode>(board, alpha, beta, currentPV, 0, i, true);
        if (timeOut)
            break;

        if (value <= alpha || value >= beta) {
            alpha = -INFINITE;
            beta = INFINITE;
            continue;
        }

        eval = value;

        if (i >= 5) {
            alpha = value - 50 * scalfac;
            beta = value + 50 * scalfac;
        }

        if (print) {
            std::string temp = std::to_string(value.value) + "  ";
            temp += " Depth:" + std::to_string(i) + " | ";
            temp += " NodeCount: " + std::to_string(nodeCounter) + "\n";

            temp += currentPV.toString();
            temp += "\n";
            temp += "\n";
            std::cout << temp;
            std::cout << "Time needed: " << (getSystemTime() - endTime + time) << "\n";
        }

        mainPV = currentPV;
        best = mainPV[0];
        ++i;
    }
    return eval;
}


template<NodeType type>
Value quiescene(Board &board, Value alpha, Value beta, Line &pv, int ply) {
    constexpr bool inPVLine = (type == PVNode);
    assert(alpha.isEval() && beta.isEval());
    nodeCounter++;
    if (ply >= MAX_PLY) {
        return board.getMover() * gameWeights.evaluate(board.getPosition());
    }

    MoveListe moves;
    getCaptures(board.getPosition(), moves);
    Value bestValue = -INFINITE;

    if (moves.isEmpty()) {

        if (board.getPosition().isWipe()) {
            return Value::loss(ply);
        }

        if (board.getPosition().hasThreat()) {
            return alphaBeta<type>(board, alpha, beta, pv, ply, 1, false);
        }



        bestValue = board.getMover() * gameWeights.evaluate(board.getPosition());
        if (bestValue >= beta) {
            return bestValue;
        }
    }

    if (inPVLine && ply < mainPV.length()) {
        moves.putFront(mainPV[ply]);
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
    assert(alpha.isEval() && beta.isEval());
    if ((nodeCounter & 16383u) == 0u && getSystemTime() >= endTime) {
        timeOut = true;
        return 0;
    }

    if (ply > 0 && board.isRepetition()) {
        return 0;
    }

    if (depth == 0) {
        return quiescene<type>(board, alpha, beta, pv, ply);
    }



    MoveListe sucessors;
    getMoves(board.getPosition(), sucessors);
    if (sucessors.isEmpty()) {
        return Value::loss(ply);
    }


    /*  if (ply ==0) {
          static std::mt19937 generator(getSystemTime());
          std::shuffle(sucessors.liste.begin(), sucessors.liste.end(), generator);
      }*/



    NodeInfo info;
#ifndef TRAIN
    TT.findHash(board.getCurrentKey(), depth, &alpha.value, &beta.value, info);
    info.value = info.value.valueFromTT(ply);
#endif


    if (!inPVLine && ply > 0 && alpha >= beta) {
        return info.value;
    }




    if (!inPVLine && prune && depth >= 5 && !beta.isWin()) {
        Value margin = (15 * scalfac * depth);
        Value newBeta = addSafe(beta, margin);
        int newDepth = (depth * 40) / 100;
        Line local;
        Value value = alphaBeta<type>(board, newBeta - 1, newBeta, local, ply+1, newDepth, false);
        if (value >= newBeta) {
            value = addSafe(value, ~margin);
            return value;
        }
    }



    if (inPVLine && ply < mainPV.length()) {
        sucessors.putFront(mainPV[ply]);
    }

    sucessors.sort(info.move, inPVLine, board.getMover());


    Value bestValue = -INFINITE;
    Move bestMove;
    Value alphaOrig = alpha;

    int extension = 0;
    if (sucessors.length() == 1 && sucessors[0].isCapture()) {
        extension += 1;
    }


    int newDepth = depth - 1 + extension;

    for (int i = 0; i < sucessors.length(); ++i) {
        bool is_promotion = sucessors[i].isPromotion(board.getPosition().K);
        board.makeMove(sucessors[i]);
        Value value;
        Line localPV;
        if (i == 0) {
            value = ~alphaBeta<type>(board, ~beta, ~alpha, localPV, ply + 1, newDepth, prune);
        } else {
            int reduce = 0;
            if (depth >= 2 && !sucessors[i].isCapture() && i > ((inPVLine) ? 3 : 1) &&
                !is_promotion) {
                reduce = 1;
                if (i >= 4 && depth>2) {
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
            bestMove = sucessors[i];
            if (value >= beta) {
                Statistics::mPicker.updateHHScore(sucessors[i], board.getMover(), depth);
                Statistics::mPicker.updateBFScore(sucessors.liste.begin(), i, board.getMover(), depth);
                break;
            }
            if (value > alpha) {
                pv.clear();
                pv.concat(bestMove, localPV);
                alpha = value;
            }
        }
    }
#ifndef TRAIN
    if (bestValue <= alphaOrig) {
        TT.storeHash(bestValue.toTT(ply), board.getCurrentKey(), TT_UPPER, depth, bestMove);
    } else if (bestValue >= beta) {
        TT.storeHash(bestValue.toTT(ply), board.getCurrentKey(), TT_LOWER, depth, bestMove);
    } else {
        TT.storeHash(bestValue.toTT(ply), board.getCurrentKey(), TT_EXACT, depth, bestMove);
    }
#endif
    return bestValue;
}