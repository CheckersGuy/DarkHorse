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


#ifndef TRAIN
Weights<short> gameWeights;
#else
Weights<double> gameWeights;
#endif

void initialize() {
#ifdef __EMSCRIPTEN__
    Bits::set_up_bitscan();
#endif
    gameWeights.loadWeights<uint32_t>("test2.weights");
    Zobrist::initializeZobrisKeys();
}


Value searchValue(Board &board, int depth, uint32_t time, bool print) {
    Move best;
    return searchValue(board, best, depth, time, print);
}

Value searchValue(Board &board, Move &best, int depth, uint32_t time, bool print) {
    Statistics::mPicker.clearScores();
    TT.age_counter++;
    nodeCounter = 0;
    mainPV.clear();
    TT.clear();
    endTime = getSystemTime() + time;
    int i = 1;
    Value eval = NONE;
    Local local;

    while (i <= depth && i <= MAX_PLY) {
        Line new_pv;
        Search::search_asp(local, new_pv, board, eval, i);
        if (!isMateVal(local.best_score) && !isEval(local.best_score))
            break;

        eval = local.best_score;
        mainPV = new_pv;
        if (print) {
            std::string temp = std::to_string(eval) + " ";
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
        if (isMateVal(local.best_score)) {
            break;
        }
    }
    return eval;
}

namespace Search {

    Depth reduce(Local &local, Board &board, Move move) {
        Depth red = 0;
        const bool is_promotion = move.isPromotion(board.getPosition().K);
        if (local.depth > 2 && !move.isCapture() && !is_promotion && local.i >= ((local.pv_node) ? 3 : 1)) {
            red = 1;
            if (!local.pv_node && local.i >= 4) {
                red = 2;
            }
        }
        return red;
    }


    Value search(Board &board, Line &pv, Value alpha, Value beta, Ply ply, Depth depth, Move skip_move, bool prune) {
        pv.clear();
        //checking time-used
        if ((nodeCounter & 16383u) == 0u && getSystemTime() >= endTime) {
            return board.getMover() * NONE;
        }
        if (board.isRepetition()) {
            return 0;
        }
        //qs
        MoveListe liste;

        getMoves(board.getPosition(), liste);


        if (liste.isEmpty()) {
            return loss(ply);
        }

        if (depth == 0) {
            return Search::qs(board, pv, alpha, beta, ply, depth);
        }


        Local local;
        local.best_score = NONE;
        local.sing_score = NONE;
        local.alpha = alpha;
        local.beta = beta;
        local.ply = ply;
        local.depth = depth;
        local.prune = prune;
        local.move = Move{};
        local.skip_move = skip_move;

        local.pv_node = (beta != alpha + 1);


        //checking win condition


        NodeInfo info;
        Move tt_move;


        uint64_t pos_key = board.getPosition().key;


        if (!local.skip_move.isEmpty()) {
            pos_key ^= Zobrist::get_move_key(board.getPosition(), local.skip_move);
        }

        if (ply >= MAX_PLY) {
            return board.getMover() * gameWeights.evaluate(board.getPosition(), ply);
        }


        // tb-probing
#ifndef TRAIN

        if (TT.findHash(pos_key, info)) {
            tt_move = (info.move_index == std::numeric_limits<uint8_t>::max()) ? Move{} : liste[info.move_index];
            auto tt_score = valueFromTT(info.score, ply);
            if (info.depth >= depth && info.score != NONE) {
                if ((info.flag == TT_LOWER && tt_score >= local.beta)
                    || (info.flag == TT_UPPER && tt_score <= local.alpha)
                    || info.flag == TT_EXACT) {
                    return tt_score;
                }
            }

            if ((info.flag == TT_LOWER && isWin(tt_score) && tt_score >= local.beta)
                || (info.flag == TT_UPPER && isLoss(tt_score) && tt_score <= local.alpha)) {
                return tt_score;
            }

            if (info.flag == TT_LOWER && info.depth >= depth - 4 && isEval(tt_score) &&
                std::find(liste.begin(), liste.end(), tt_move) != liste.end()) {
                local.sing_score = tt_score;
                local.sing_move = tt_move;
            }
        }
#endif
        //probcut






        if (!local.pv_node && local.prune && local.depth >= 5 && isEval(local.beta)) {
            Value margin = (15 * local.depth);
            Value newBeta = local.beta + margin;
            Depth newDepth = (depth * 40) / 100;
            Line new_pv;
            Value value = Search::search(board, new_pv, newBeta - 1, newBeta, ply + 1, newDepth, Move{}, false);
            if (value >= newBeta) {
                value = value - margin;
                pv = new_pv;
                return value;
            }
        }



        if (local.pv_node && ply < mainPV.length()) {
            liste.putFront(mainPV[ply]);
        }

        //sorting
        liste.sort(tt_move, local.pv_node, board.getMover());

        //move-loop
        Search::move_loop(local, board, pv, liste);


        //updating search stats
        if (local.best_score > local.alpha && liste.length() > 1) {
            Statistics::mPicker.update_scores(&liste.liste[0], local.move, board.getMover(),
                                              depth);
        }
#ifndef TRAIN
        //storing tb-entries
        Value tt_value = toTT(local.best_score, ply);
        Flag flag;
        if (local.best_score > local.alpha) {
            flag = TT_LOWER;
        } else if (local.best_score < local.beta) {
            flag = TT_UPPER;
        } else {
            flag = TT_EXACT;
        }

        {
            uint8_t tt_m = std::numeric_limits<uint8_t>::max();
            if (local.best_score > local.alpha) {
                auto index = liste.get_move_index(local.move);
                tt_m = (index.has_value()) ? index.value() : tt_m;
            }

            TT.storeHash(tt_value, board.getPosition(), flag, depth, tt_m);
        }
#endif
        return local.best_score;
    }

    Value qs(Board &board, Line &pv, Value alpha, Value beta, Ply ply, Depth depth) {
        bool in_pv_line = (beta != alpha + 1);
        nodeCounter++;
        pv.clear();

        if (ply >= MAX_PLY) {
            return board.getMover() * gameWeights.evaluate(board.getPosition(), ply);
        }


        MoveListe moves;
        getCaptures(board.getPosition(), moves);
        Value bestValue = NONE;

        if (moves.isEmpty()) {
            if (depth == 0 && board.getPosition().hasThreat()) {
                return Search::search(board, pv, alpha, beta, ply + 1, 1, Move{},
                                      false);
            }
            bestValue = board.getMover() * gameWeights.evaluate(board.getPosition(), ply);
            if (bestValue >= beta) {
                return bestValue;
            }
        }

        if (in_pv_line && ply < mainPV.length()) {
            moves.putFront(mainPV[ply]);
        }
        for (int i = 0; i < moves.length(); ++i) {
            Line localPV;
            board.makeMove(moves.liste[i]);
            Value value = -Search::qs(board, localPV, -beta, -std::max(alpha, bestValue), ply + 1, depth - 1);
            board.undoMove();
            if (value > bestValue) {
                bestValue = value;
                pv.concat(moves[i], localPV);
                if (value >= beta)
                    break;

            }
        }

        return bestValue;
    }


    Value searchMove(Move move, Local &local, Board &board, Line &line, int extension) {
        Depth reduction = Search::reduce(local, board, move);
        //singular move extension




       if (local.pv_node
            && local.depth >= 7
            && move == local.sing_move
            && local.skip_move.isEmpty()
            && extension == 0
                ) {
            constexpr Value margin = 15;
            Value new_alpha = local.sing_score - margin;
            Line new_pv;
            Value value = Search::search(board, new_pv, new_alpha , new_alpha+1, local.ply, local.depth - 4, move,
                                         local.prune);


            if (value <= new_alpha)
                extension = 1;

        }



        Depth new_depth = local.depth - 1 + extension;

        Value new_alpha = std::max(local.alpha, local.best_score);

        board.makeMove(move);
        Value val;

        if ((local.pv_node && local.i != 0) || reduction != 0) {
            val = -Search::search(board, line, -new_alpha - 1, -new_alpha, local.ply + 1, new_depth - reduction, Move{},
                                  local.prune);
            if (val > new_alpha) {
                val = -Search::search(board, line, -local.beta, -new_alpha, local.ply + 1, new_depth, Move{},
                                      local.prune);
            }
        } else {
            val = -Search::search(board, line, -local.beta, -new_alpha, local.ply + 1, new_depth, Move{},
                                  local.prune);
        }

        board.undoMove();
        return val;

    }

    void move_loop(Local &local, Board &board, Line &pv, MoveListe &liste) {
        local.i = 0;
        const auto num_moves = liste.length();
        int extension = (liste.length() == 1) ? 1 : 0;
        while (local.best_score < local.beta && local.i < num_moves) {
            Move move = liste[local.i];
            if (move != local.skip_move) {
                Line local_pv;
                Value value = searchMove(move, local, board, local_pv, extension);
                if (value > local.best_score) {
                    local.move = move;
                    local.best_score = value;
                    pv.concat(move, local_pv);
                }
            }

            local.i++;
        }

    }


    void search_root(Local &local, Line &line, Board &board, Value alpha, Value beta, Depth depth) {
        line.clear();
        local.best_score = NONE;
        local.sing_score = NONE;
        local.alpha = alpha;
        local.beta = beta;
        local.ply = 0;
        local.depth = depth;
        local.skip_move = Move{};
        local.sing_move = Move{};
        local.move = Move{};
        local.prune = true;
        local.pv_node = (beta != alpha + 1);


        MoveListe liste;
        getMoves(board.getPosition(), liste);
        liste.putFront(mainPV[0]);
        liste.sort(Move{}, true, board.getMover());

        move_loop(local, board, line, liste);


    }

    void search_asp(Local &local, Line &line, Board &board, Value last_score, Depth depth) {
    /*    if (depth >= 5 && isEval(last_score)) {
            Value margin = 10;
            Value alpha_margin = margin;
            Value beta_margin = margin;

            while (std::max(alpha_margin, beta_margin) < 500) {
                Value alpha = last_score - alpha_margin;
                Value beta = last_score + beta_margin;

                search_root(local, line, board, alpha, beta, depth);

                Value score = local.best_score;
                if (score <= alpha) {
                    alpha_margin *= 2;
                } else if (score >= beta) {
                    beta_margin *= 2;
                } else {
                    return;
                }
            }
        }
*/
        search_root(local, line, board, -INFINITE, INFINITE, depth);

    }


}
