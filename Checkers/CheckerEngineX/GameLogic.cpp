//
// Created by Robin on 10.06.2017.
//

#include "GameLogic.h"


uint64_t nodeCounter = 0;
Line mainPV;
Value last_evaluation = INFINITE;

extern char *output;
bool timeOut = false;
uint64_t endTime = 1000000000;

void setHashSize(uint32_t hash) {
    TT.resize(hash);
}


#ifdef TRAIN
Weights<double> gameWeights;
#else
Weights<short> gameWeights;
#endif


void initialize() {
#ifdef __EMSCRIPTEN__
    Bits::set_up_bitscan();
#endif
    gameWeights.loadWeights<uint32_t>("checkers8.weights");
    Zobrist::initializeZobrisKeys();
}


Value searchValue(Board &board, int depth, uint32_t time, bool print) {
    Move best;
    return searchValue(board, best, depth, time, print);
}

Value searchValue(Board &board, Move &best, int depth, uint32_t time, bool print) {
    return searchValue(board, -INFINITE, INFINITE, best, depth, time, print);
}

Value searchValue(Board &board, Value alpha, Value beta, Move &best, int depth, uint32_t time, bool print) {
    Statistics::mPicker.clearScores();
    nodeCounter = 0;
    mainPV.clear();
    TT.clear();
    timeOut = false;
    endTime = getSystemTime() + time;
    int i = 1;
    Value value;
    Value eval = DRAW;
    Local local;
    local.board = board;
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
            alpha = value - 3 * scalfac;
            beta = value + 3 * scalfac;
        }

        if (print) {
            std::string temp = std::to_string(value) + "  ";
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
    constexpr bool in_pv = type == PVNode;
    nodeCounter++;
    if (ply >= MAX_PLY) {
        return board.getMover() * gameWeights.evaluate(board.getPosition());
    }
    pv.clear();
    MoveListe moves;
    getCaptures(board.getPosition(), moves);
    Value bestValue = -INFINITE;

    if (moves.isEmpty()) {

        if (board.getPosition().isWipe()) {
            return loss(ply);
        }
        //loss-distance pruning
        if (loss(ply + 2) >= beta) {
            return loss(ply + 2);
        }

        //threat-detection -> 1 ply search
        if (board.getPosition().hasThreat()) {
            return alphaBeta<type>(board, alpha, beta, pv, ply, 1, false);
        }

        bestValue = board.getMover() * gameWeights.evaluate(board.getPosition());
        if (bestValue >= beta) {
            return bestValue;
        }
    }

    if (in_pv && ply < mainPV.length()) {
        moves.putFront(mainPV[ply]);
    }

    for (int i = 0; i < moves.length(); ++i) {
        Line localPV;
        board.makeMove(moves.liste[i]);;
        Value value;
        if (i == 0) {
            value = -quiescene<type>(board, -beta, -alpha, localPV, ply + 1);
        } else {
            value = -quiescene<NONPV>(board, -beta, -alpha, localPV, ply + 1);
        }
        board.undoMove();
        if (value > bestValue) {
            bestValue = value;
            if (value >= beta)
                break;

            if (value > alpha) {
                alpha = value;
                pv.concat(moves[i], localPV);
            }

        }
    }

    return bestValue;
}

template<NodeType type>
Value
alphaBeta(Board &board, Value alpha, Value beta, Line &pv, int ply, int depth, bool prune) {
    pv.clear();
    const bool inPVLine = type == PVNode;
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
        return loss(ply);
    }


    NodeInfo info;


#ifndef TRAIN
    if (ply > 0 && TT.findHash(board.getPosition(), info)) {
        auto tt_score = valueFromTT(info.score, ply);
        if (info.depth >= depth) {
            if ((info.flag == TT_LOWER && info.score >= beta)
                || (info.flag == TT_UPPER && info.score <= alpha)
                || info.flag == TT_EXACT) {
                return tt_score;
            }
        }

        if ((info.flag == TT_LOWER && isWin(info.score) && info.score >= beta)
            || (info.flag == TT_UPPER && isLoss(info.score) && info.score <= alpha)) {
            return tt_score;
        }
    }
#endif

    if (!inPVLine && prune && depth >= 3 && isEval(beta)) {
        Value margin = (10 * scalfac * depth);
        Value newBeta = addSafe(beta, margin);
        int newDepth = (depth * 40) / 100;
        Line local;
        Value value = alphaBeta<type>(board, newBeta - 1, newBeta, local, ply, newDepth, false);
        if (value >= newBeta) {
            value = addSafe(value, -margin);
            return value;
        }
    }


    if (inPVLine && ply < mainPV.length()) {
        sucessors.putFront(mainPV[ply]);
    }


#ifdef GENERATOR
    if (ply == 0) {
        static std::mt19937_64 generator(getSystemTime());
        auto next = sucessors.liste.begin();
        std::advance(next, sucessors.length());
        std::shuffle(sucessors.liste.begin(), next, generator);
    }
#endif


    sucessors.sort(info.move, inPVLine, board.getMover());

    Value bestValue = -INFINITE;
    Move bestMove;
    Value alphaOrig = alpha;

    int extension = 0;
    if (sucessors.length() == 1) {
        extension = 1;
    }


    int newDepth = depth - 1 + extension;

    for (int i = 0; i < sucessors.length(); ++i) {
        bool is_promotion = sucessors[i].isPromotion(board.getPosition().K);
        board.makeMove(sucessors[i]);
        Value value;
        Line localPV;
        if (i == 0) {
            value = -alphaBeta<type>(board, -beta, -alpha, localPV, ply + 1, newDepth, prune);
        } else {
            int reduce = 0;
            if (depth >= 2 && !sucessors[i].isCapture() && i > ((inPVLine) ? 3 : 1) &&
                !is_promotion) {
                reduce = 1;
                if (i >= 4 && depth > 2) {
                    reduce = 2;
                }
            }

            value = -alphaBeta<NONPV>(board, -alpha - 1, -alpha, localPV, ply + 1, newDepth - reduce, prune);
            if (value > alpha && value < beta) {
                value = -alphaBeta<NONPV>(board, -beta, -alpha, localPV, ply + 1, newDepth, prune);
            }
        }
        board.undoMove();
        if (value > bestValue) {
            bestValue = value;
            bestMove = sucessors[i];
            if (value >= beta) {
                if (sucessors.length() > 1u)
                    Statistics::mPicker.update_scores(&sucessors[0], i, board.getMover(), depth);
                break;
            }

            if (value > alpha) {
                pv.concat(bestMove, localPV);
                alpha = bestValue;
            }


        }
    }
#ifndef TRAIN
    Value tt_value = toTT(bestValue, ply);
    if (bestValue <= alphaOrig) {
        TT.storeHash(tt_value, board.getPosition(), TT_UPPER, depth, bestMove);
    } else if (bestValue >= beta) {
        TT.storeHash(tt_value, board.getPosition(), TT_LOWER, depth, bestMove);
    } else {
        TT.storeHash(tt_value, board.getPosition(), TT_EXACT, depth, bestMove);
    }

#endif
    return bestValue;
}

namespace Search {

    Depth reduce(Local &local, Move move, bool in_pv_line) {
        Depth red = 0;
        const bool is_promotion = move.isPromotion(local.board.getPosition().K);
        if (local.depth >= 2 && !move.isCapture() && !is_promotion && local.i > ((in_pv_line) ? 3 : 1)) {
            red = 1;
            if (local.i >= 4 && local.depth > 2) {
                red = 2;
            }
        }
        return red;
    }


    template<NodeType type>
    Value search(Local &local, Line &line, Value alpha, Value beta, Ply ply, Depth depth, bool prune) {
        constexpr bool in_pv_line = (type == PVNode);
        local.best_score = -INFINITE;
        local.alpha = alpha;
        local.beta = beta;
        local.ply = ply;
        local.depth = depth;
        local.i = 0;
        local.prune = prune;
        local.move = Move{};
        local.skip_move = Move{};
        line.clear();

        //checking time-used
        if ((local.node_counter & 16383u) == 0u && getSystemTime() >= endTime) {
            timeOut = true;
            return 0;
        }
        //Repetition check
        if (local.ply > 0 && local.board.isRepetition()) {
            return 0;
        }

        //qs
        if (local.depth == 0) {
            return 0;
        }

        MoveListe liste;

        getMoves(local.board.getPosition(), liste);
        //checking win condition
        if (liste.isEmpty()) {
            return loss(ply);
        }

        NodeInfo info;
        Value alphaOrig = alpha;

        // tb-probing
        if (local.ply > 0 && TT.findHash(local.board.getPosition(), info)) {
            auto tt_score = valueFromTT(info.score, local.ply);
            if (info.depth >= local.depth) {
                if ((info.flag == TT_LOWER && info.score >= local.beta)
                    || (info.flag == TT_UPPER && info.score <= local.alpha)
                    || info.flag == TT_EXACT) {
                    return tt_score;
                }
            }

            if ((info.flag == TT_LOWER && isWin(info.score) && info.score >= local.beta)
                || (info.flag == TT_UPPER && isLoss(info.score) && info.score <= local.alpha)) {
                return tt_score;
            }
            //here would go the code for  doing singular move extensions
            /*if (info.flag == TT_LOWER && info.depth - 4 >= local.depth &&
                std::find(liste.begin(), liste.end(), info.move) != liste.end()) {
                local.sing_score = info.score;
                local.sing_move = info.move;
            }*/
        }
        //probcut
        if (!in_pv_line && local.prune && local.depth >= 3 && isEval(local.beta)) {
            Value margin = (10 * scalfac * local.depth);
            Value newBeta = addSafe(local.beta, margin);
            Depth newDepth = (local.depth * 40) / 100;
            Line new_pv;
            Value value = Search::search<type>(local, new_pv, newBeta, newBeta - 1, local.ply, newDepth, false);
            if (value >= newBeta) {
                value = addSafe(value, -margin);
                return value;
            }
        }

        //move-loop

        //jump extension
        if (liste.length() == 1 && liste[0].isCapture()) {
            //needs to be fixed
        }

        //sorting
        liste.sort(info.move, in_pv_line, local.board.getMover());

        Search::move_loop<type>(local, liste, line);

        //updating search stats
        if (local.best_score >= local.beta) {
            if (liste.length() > 1) {
                Statistics::mPicker.update_scores(liste.liste.begin(), local.i, local.board.getMover(),
                                                  local.depth);
            }
        }

        //storing tb-entries
        Value tt_value = toTT(local.best_score, ply);
        if (local.best_score <= alphaOrig) {
            TT.storeHash(tt_value, local.board.getPosition(), TT_UPPER, depth, local.move);
        } else if (local.best_score >= beta) {
            TT.storeHash(tt_value, local.board.getPosition(), TT_LOWER, depth, local.move);
        } else {
            TT.storeHash(tt_value, local.board.getPosition(), TT_EXACT, depth, local.move);
        }


        return local.best_score;
    }


    template<NodeType type>
    Value qs(Local &local, Line &line, Ply ply) {
        constexpr bool in_pv = type == PVNode;
        nodeCounter++;
        if (ply >= MAX_PLY) {
            return local.board.getMover() * gameWeights.evaluate(local.board.getPosition());
        }
        line.clear();
        MoveListe moves;
        getCaptures(local.board.getPosition(), moves);
        Value bestValue = -INFINITE;

        if (moves.isEmpty()) {

            if (local.board.getPosition().isWipe()) {
                return loss(ply);
            }
            //loss-distance pruning
            if (loss(ply + 2) >= local.beta) {
                return loss(ply + 2);
            }

            //threat-detection -> 1 ply search
            if (local.board.getPosition().hasThreat()) {
                return Search::search<type>(local, line, local.alpha, local.beta, local.ply, 1,
                                            false);
            }

            bestValue = local.board.getMover() * gameWeights.evaluate(local.board.getPosition());
            if (bestValue >= local.beta) {
                return bestValue;
            }
        }

        if (in_pv && ply < mainPV.length()) {
            moves.putFront(mainPV[ply]);
        }

        for (int i = 0; i < moves.length(); ++i) {
            Line localPV;
            local.board.makeMove(moves.liste[i]);;
            Value value;
            if (i == 0) {
                value = -Search::qs<type>(local, line, ply + 1);
            } else {
                value = -Search::qs<NONPV>(local, line, ply + 1);
            }
            local.board.undoMove();
            if (value > bestValue) {
                bestValue = value;
                if (value >= local.beta)
                    break;
                if (value > local.alpha) {
                    line.concat(moves[i], localPV);
                }

            }
        }

        return bestValue;
    }

    template<NodeType type>
    void searchMove(Move move, Local &local, Line &line) {
        constexpr bool in_pv_line = (type == PVNode);
        //everything that is specific to a move goes into search_move
        //that includes reductions and extensions (lmr and probuct and jump extension)
        Depth reduction = Search::reduce(local, move, in_pv_line);
        Depth new_depth = local.depth - reduction - 1;
        const bool is_first_move = local.i == 0;

        //singular move extension

        /*      if (local.skip_move.isEmpty() && extension == 0 && local.depth >= 8 && move == local.sing_move) {
                  //there will be some other conditions added
                  constexpr Value margin = 40;
                  Value new_alpha = local.sing_score - margin;
                  Line new_pv;
                  Value value = Search::search<type>(local, new_pv, new_alpha, new_alpha + 1, local.ply, local.depth - 4,
                                                     local.prune);
                  if (value <= local.alpha)
                      extension = 1;
              }*/

        //making the move
        local.board.makeMove(move);

        Line new_pv;
        Value val;
        if (is_first_move) {
            //doing a fully window search
            val = -Search::search<type>(local, new_pv, -local.beta, -local.alpha, local.ply + 1, new_depth,
                                        local.prune);
        } else {
            //zero-window and research
            val = -Search::search<NONPV>(local, new_pv, -local.alpha - 1, -local.alpha, local.ply + 1, new_depth,
                                         local.prune);
            //doing the research if we didnt fail low
            if (val > local.alpha && val < local.beta) {
                val = -Search::search<NONPV>(local, new_pv, -local.beta, -local.alpha, local.ply + 1, new_depth,
                                             local.prune);
            }

        }

        //undoing the move
        local.board.undoMove();

        if (val > local.best_score) {
            local.best_score = val;
            local.move = move;
            if (val >= local.beta) {
                return;
            }
            if (val > local.alpha) {
                line.concat(local.move, new_pv);
                local.alpha = local.best_score;
            }
        }

    }

    template<NodeType type>
    void move_loop(Local &local, const MoveListe &liste, Line &line) {
        //move-loop goes here
        //skip-move and so on
        local.i = 0;
        while (local.best_score < local.beta && local.i < liste.length()) {
            Move move = liste[local.i++];
            searchMove<type>(move, local, line);
        }

    }


    void search_root(Local &local, Value alpha, Value beta, Depth depth) {
        //root search can throw out a couple of things
        //no prob-cut or probing of the hash_table
        //no quiescent search


        local.best_score = -INFINITE;
        local.alpha = alpha;
        local.beta = beta;
        local.ply = 0;
        local.depth = depth;
        local.i = 0;
        local.skip_move = Move{};
        local.sing_move = Move{};
        local.move = Move{};
        local.prune = true;
        local.pv_line.clear();

        //little bit more work on this

        //generating the moves

        //other things will go here too

        MoveListe liste;
        getMoves(local.board.getPosition(), liste);

        move_loop<PVNode>(local, liste, local.pv_line);

        //storing tb-entries
        Value tt_value = toTT(local.best_score, local.ply);
        if (local.best_score <= alpha) {
            TT.storeHash(tt_value, local.board.getPosition(), TT_UPPER, depth, local.move);
        } else if (local.best_score >= beta) {
            TT.storeHash(tt_value, local.board.getPosition(), TT_LOWER, depth, local.move);
        } else {
            TT.storeHash(tt_value, local.board.getPosition(), TT_EXACT, depth, local.move);
        }


    }


}