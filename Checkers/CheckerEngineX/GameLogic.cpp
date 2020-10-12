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
    endTime = getSystemTime() + time;
    int i = 1;
    Value eval = -INFINITE - 1;
    Local local;
    while (i <= depth && i <= MAX_PLY) {
        Line new_pv;
        Search::search_asp(local, new_pv, board, eval, i);
        if (std::abs(local.best_score) == TIME_OUT)
            break;

        eval = local.best_score;
        mainPV = new_pv;
        if (print) {
            std::string temp = std::to_string(eval) + "  ";
            temp += " Depth:" + std::to_string(i) + " | ";
            temp += " NodeCount: " + std::to_string(nodeCounter) + "\n";
            temp += mainPV.toString();
            temp += "\n";
            temp += "\n";
            std::cout << temp;
            std::cout << "Time needed: " << (getSystemTime() - endTime + time) << "\n";
        }


        best = mainPV.getFirstMove();
        ++i;
    }
    return eval;
}

namespace Search {

    Depth reduce(Local &local, Board &board, Move move, bool in_pv_line) {
        Depth red = 0;
        const bool is_promotion = move.isPromotion(board.getPosition().K);
        if (local.depth >= 2 && !move.isCapture() && !is_promotion && local.i >= ((in_pv_line) ? 3 : 1)) {
            red = 1;
            if (!in_pv_line && local.i >= 4 && local.depth > 2) {
                red = 2;
            }
        }
        return red;
    }


    template<NodeType type>
    Value search(Board &board, Line &pv, Value alpha, Value beta, Ply ply, Depth depth, bool prune) {
        constexpr bool in_pv_line = (type == PVNode);
        pv.clear();
        //checking time-used
        if ((nodeCounter & 16383u) == 0u && getSystemTime() >= endTime) {
            return board.getMover() * TIME_OUT;
        }
        //Repetition check
        if (ply > 0 && board.isRepetition()) {
            return 0;
        }

        //qs
        if (depth == 0) {
            return Search::qs<type>(board, pv, alpha, beta, ply);
        }
        Local local;
        local.best_score = -INFINITE;
        local.alpha = alpha;
        local.beta = beta;
        local.ply = ply;
        local.depth = depth;
        local.prune = prune;
        local.move = Move{};
        local.skip_move = Move{};

        MoveListe liste;

        getMoves(board.getPosition(), liste);
        //checking win condition
        if (liste.isEmpty()) {
            return loss(ply);
        }

        NodeInfo info;
        Value alphaOrig = local.alpha;

        // tb-probing
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
            //here would go the code for  doing singular move extensions
            if (info.flag == TT_LOWER && info.depth - 4 >= depth &&
                std::find(liste.begin(), liste.end(), info.move) != liste.end()) {
                local.sing_score = info.score;
                local.sing_move = info.move;
            }
        }
        //probcut

        if (!in_pv_line && prune && depth >= 5 && isEval(beta)) {
            Value margin = (20 * scalfac * depth);
            Value newBeta = beta + margin;
            Depth newDepth = (depth * 40) / 100;
            Line new_pv;
            Value value = Search::search<type>(board, new_pv, newBeta - 1, newBeta, ply, newDepth, false);
            if (value >= newBeta) {
                value = value - margin;
                return value;
            }
            pv = new_pv;
        }


        if (in_pv_line && ply < mainPV.length()) {
            liste.putFront(mainPV[ply]);
        }

        //sorting
        liste.sort(info.move, in_pv_line, board.getMover());
        //move-loop
        Search::move_loop<type>(local, board, pv, liste);




        //updating search stats
        if (local.best_score >= local.beta) {
            Statistics::mPicker.update_scores(&liste.liste[0], local.move, board.getMover(),
                                              local.depth);
        }

        //storing tb-entries
        Value tt_value = toTT(local.best_score, ply);
        if (local.best_score <= alphaOrig) {
            TT.storeHash(tt_value, board.getPosition(), TT_UPPER, depth, local.move);
        } else if (local.best_score >= beta) {
            TT.storeHash(tt_value, board.getPosition(), TT_LOWER, depth, local.move);
        } else {
            TT.storeHash(tt_value, board.getPosition(), TT_EXACT, depth, local.move);
        }

        return local.best_score;
    }


    template<NodeType type>
    Value qs(Board &board, Line &pv, Value alpha, Value beta, Ply ply) {
        constexpr bool in_pv_line = type == PVNode;
        nodeCounter++;
        pv.clear();
        if (ply >= MAX_PLY) {
            return board.getMover() * gameWeights.evaluate(board.getPosition());
        }

        MoveListe moves;
        getCaptures(board.getPosition(), moves);
        Value bestValue = -INFINITE;

        if (moves.isEmpty()) {

            if (board.getPosition().isWipe()) {
                return loss(ply);
            }
            //loss-distance pruning
            /*    if (loss(ply + 2) >= beta) {
                    return loss(ply + 2);
                }
    */
            //threat-detection -> 1 ply search
            if (board.getPosition().hasThreat()) {
                return Search::search<type>(board, pv, alpha, beta, ply, 1,
                                            false);
            }

            bestValue = board.getMover() * gameWeights.evaluate(board.getPosition());
            if (bestValue >= beta) {
                return bestValue;
            }
        }

        if (in_pv_line && ply < mainPV.length()) {
            moves.putFront(mainPV[ply]);
        }
        for (int i = 0; i < moves.length(); ++i) {
            Line localPV;
            board.makeMove(moves.liste[i]);;
            Value value;
            if (i == 0) {
                value = -Search::qs<type>(board, localPV, -beta, -alpha, ply + 1);
            } else {
                value = -Search::qs<NONPV>(board, localPV, -beta, -alpha, ply + 1);
            }
            board.undoMove();
            if (value > bestValue) {
                bestValue = value;
                pv.concat(moves[i], localPV);
                if (value >= beta)
                    break;
                if (value > alpha) {
                    alpha = value;
                }

            }
        }

        return bestValue;
    }

    template<NodeType type>
    Value searchMove(Move move, Local &local, Board &board, Line &line, int extension) {
        constexpr bool in_pv_line = (type == PVNode);
        //everything that is specific to a move goes into search_move
        //that includes reductions and extensions (lmr and probuct and jump extension)
        Depth reduction = Search::reduce(local, board, move, in_pv_line);



        //singular move extension

        if (local.skip_move.isEmpty() && extension == 0 && local.depth >= 8 && move == local.sing_move) {
            //there will be some other conditions added
            constexpr Value margin = 40 * scalfac;
            Value new_alpha = local.sing_score - margin;
            Line new_pv;
            Value value = Search::search<type>(board, new_pv, new_alpha, new_alpha + 1, local.ply, local.depth - 4,
                                               local.prune);
            if (value <= local.alpha)
                extension = 1;
        }

        Depth new_depth = local.depth - 1;

        Value new_alpha = std::max(local.alpha, local.best_score);

        board.makeMove(move);
        Value val;

        if ((in_pv_line && local.i != 0) || reduction != 0) {
            val = -Search::search<NONPV>(board, line, -new_alpha - 1, -new_alpha, local.ply + 1, new_depth - reduction,
                                         local.prune);
            if (val > new_alpha) {
                val = -Search::search<NONPV>(board, line, -local.beta, -new_alpha, local.ply + 1, new_depth,
                                             local.prune);
            }
        } else {
            val = -Search::search<type>(board, line, -local.beta, -local.alpha, local.ply + 1, new_depth,
                                        local.prune);
        }

        board.undoMove();
        return val;

    }

    template<NodeType type>
    void move_loop(Local &local, Board &board, Line &pv, MoveListe &liste) {
        local.i = 0;
        const auto num_moves = liste.length();
        int extension = (liste.length() == 1) ? 1 : 0;
        while (local.i < num_moves) {
            Move move = liste[local.i];

            if (move != local.skip_move) {
                Line local_pv;
                Value value = searchMove<type>(move, local, board, local_pv,extension);
                if (value > local.best_score) {
                    local.best_score = value;
                    local.move = move;
                    pv.concat(move, local_pv);

                    if (value >= local.beta) {
                        break;
                    }
                    if (value > local.alpha) {
                        local.alpha = local.best_score;
                    }
                }
            }

            local.i++;
        }

    }


    void search_root(Local &local, Line &line, Board &board, Value alpha, Value beta, Depth depth) {
        line.clear();
        local.best_score = -INFINITE;
        local.alpha = alpha;
        local.beta = beta;
        local.ply = 0;
        local.depth = depth;
        local.skip_move = Move{};
        local.sing_move = Move{};
        local.move = Move{};
        local.prune = true;
        //little bit more work on this

        //generating the moves

        //other things will go here too

        MoveListe liste;
        getMoves(board.getPosition(), liste);
        liste.putFront(mainPV[0]);
        liste.sort(Move{}, true, board.getMover());
        move_loop<PVNode>(local, board, line, liste);

        if (local.best_score == TIME_OUT)
            std::cout << "TEST" << std::endl;

    }

    void search_asp(Local &local, Line &line, Board &board, Value last_score, Depth depth) {
        if (depth >= 5 && isEval(last_score)) {
            Value margin = 7 * scalfac;
            Value alpha_margin = margin;
            Value beta_margin = margin;

            while (std::max(alpha_margin, beta_margin) < 100 * scalfac) {
                Value alpha = last_score - alpha_margin;
                Value beta = last_score + beta_margin;

                search_root(local, line, board, alpha, beta, depth);

                Value score = local.best_score;
                if (std::abs(score) == TIME_OUT) {
                    break;
                } else if (score <= alpha) {
                    alpha_margin *= 2;
                } else if (score >= beta) {
                    beta_margin *= 2;
                } else {
                    return;
                }
            }
        }

        search_root(local, line, board, -INFINITE, INFINITE, depth);

    }


}