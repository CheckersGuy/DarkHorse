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
    gameWeights.loadWeights("/home/robin/DarkHorse/Training/cmake-build-debug/failSave.weights");
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
    getMoves(board.getPosition(), easyMoves);
    if (easyMoves.length() == 1) {
        best = easyMoves[0];
        return EASY_MOVE;
    }
    mainPV.clear();
    TT.clear();
    timeOut = false;
    endTime = getSystemTime() + time;
    int i = 1;

    Value alpha = -INFINITE;
    Value beta = INFINITE;
    Value gameValue;


    while (i <= depth && i <= MAX_PLY) {
        Line currentPV;
        Value value = alphaBeta<PVNode>(board, alpha, beta, currentPV, 0, i * ONE_PLY, true);
        if (timeOut)
            break;


        if (value <= alpha || value >= beta) {
            alpha = -INFINITE;
            beta = INFINITE;
            continue;
        }


        if (i >= 5) {
            alpha = value - 50*scalfac ;
            beta = value + 50*scalfac ;
        }

        if (print) {
            std::string temp = std::to_string(gameValue.value) + "  ";
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
        gameValue = value;

        ++i;
    }
    return gameValue;
}

template<NodeType type>
Value quiescene(Board &board, Value alpha, Value beta, Line &pv, int ply) {
    constexpr bool inPVLine = (type == PVNode);
    assert(alpha.isEval() && beta.isEval());
    nodeCounter++;
    MoveListe moves;
    getCaptures(board.getPosition(), moves);
    Value bestValue = -INFINITE;

    if (moves.isEmpty()) {
        if (board.getPosition().hasThreat()) {
            return alphaBeta<type>(board, alpha, beta, pv, ply, ONE_PLY, false);
        }

        if (board.getPosition().isWipe()) {
            return Value::loss(board.getMover(), ply);
        }

        bestValue = board.getMover() * gameWeights.evaluate(board.getPosition());
        if (bestValue >= beta) {
            return bestValue;
        }
    }

    if (inPVLine && ply < mainPV.length()) {
        moves.putFront(mainPV[0]);
    }

    for (int i = 0; i < moves.length(); ++i) {
        Line localPV;
        board.makeMove(moves[i]);
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

    if (ply > MAX_PLY) {
        return gameWeights.evaluate(board.getPosition()) * board.getMover();
    }

    if (depth < ONE_PLY) {
        return quiescene<type>(board, alpha, beta, pv, ply);
    }

    MoveListe sucessors;
    getMoves(board.getPosition(), sucessors);
    if (sucessors.isEmpty()) {
        return Value::loss(board.getMover(), ply);
    }


    /*if (isRoot) {
        static std::mt19937 generator(getSystemTime());
        std::shuffle(sucessors.begin(), sucessors.end(), generator);
    }
*/

    NodeInfo info;
#ifndef TRAIN
    if (ply > 0) {
        TT.findHash(board.getCurrentKey(), depth / ONE_PLY, &alpha.value, &beta.value, info);
        info.value = info.value.valueFromTT(ply);
    };
#endif


    if (!inPVLine && ply > 0 && alpha >= beta) {
        return info.value;
    }


    if (!inPVLine && prune && depth >= 3 * ONE_PLY) {
        Value margin = (5 *scalfac* depth) / ONE_PLY;
        Value newBeta = addSafe(beta, margin);
        int newDepth = (depth * 40) / 100;
        Line local;
        Value value = alphaBeta<type>(board, newBeta - 1, newBeta, local, ply, newDepth, false);
        if (value >= newBeta) {
            return addSafe(value, ~margin);
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
        extension += 500;
    }

    int newDepth = depth - ONE_PLY + extension;

    for (int i = 0; i < sucessors.length(); ++i) {
        board.makeMove(sucessors[i]);
        Value value;
        Line localPV;
        if (i == 0) {
            value = ~alphaBeta<type>(board, ~beta, ~alpha, localPV, ply + 1, newDepth, prune);
        } else {
            int reduce = 0;
            if (depth > 2 * ONE_PLY && !sucessors[i].isCapture() && i > ((inPVLine) ? 3 : 1) &&
                !sucessors[i].isPromotion()) {
                reduce = ONE_PLY;
                if (i > 3) {
                    reduce = 2000;
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
                Statistics::mPicker.updateHHScore(sucessors[i], board.getMover(), depth / ONE_PLY);
                Statistics::mPicker.updateBFScore(sucessors.begin(), i, board.getMover(), depth / ONE_PLY);
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
        TT.storeHash(bestValue.toTT(ply), board.getCurrentKey(), TT_UPPER, depth / ONE_PLY, bestMove);
    } else if (bestValue >= beta) {
        TT.storeHash(bestValue.toTT(ply), board.getCurrentKey(), TT_LOWER, depth / ONE_PLY, bestMove);
    } else {
        TT.storeHash(bestValue.toTT(ply), board.getCurrentKey(), TT_EXACT, depth / ONE_PLY, bestMove);
    }
#endif
    return bestValue;
}
