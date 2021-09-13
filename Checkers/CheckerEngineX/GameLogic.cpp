#include "GameLogic.h"


Line mainPV;
uint64_t endTime = 1000000000;
uint64_t nodeCounter = 0u;

#ifndef TRAIN
Weights<int16_t> gameWeights;
#else
Weights<double> gameWeights;
#endif


SearchGlobal glob;
Network network;
bool u_classical = false;

void initialize() {
    gameWeights.loadWeights<uint32_t>("../Training/Engines/bloomcloud.weights");
    Zobrist::initializeZobrisKeys();
    //Statistics::mPicker.init();
}

void initialize(uint64_t seed) {
    gameWeights.loadWeights<uint32_t>("../Training/Engines/currenttest14.weights");
    Zobrist::initializeZobrisKeys(seed);
    //Statistics::mPicker.init();
}

Value searchValue(Board &board, int depth, uint32_t time, bool print) {
    Move best;
    return searchValue(board, best, depth, time, print);
}

void use_classical(bool flag) {
    u_classical = flag;
}


Value searchValue(Board board, Move &best, int depth, uint32_t time, bool print) {
    Statistics::mPicker.clearScores();
    glob.sel_depth = 0u;
    TT.age_counter++;
    nodeCounter = 0;
    mainPV.clear();
    //TT.clear();
    endTime = getSystemTime() + time;
    Value eval = INFINITE;
    Local local;


    if (depth == 0) {
        //returning q-search
        return Search::qs(false, board, mainPV, -INFINITE, INFINITE, 0, 0);
    }

    for (int i = 1; i <= depth; i += 2) {
        try {
            Search::search_asp(local, board, eval, i);
        } catch (std::string &msg) {
            break;
        }

        if (!isMateVal(local.best_score) && !isEval(local.best_score))
            break;

        eval = local.best_score;
        best = mainPV.getFirstMove();
        if (print) {
            std::string temp = std::to_string(eval) + " ";
            temp += " Depth:" + std::to_string(i) + " | " + std::to_string(glob.sel_depth) + " | ";
            temp += " NodeCount: " + std::to_string(nodeCounter) + "\n";
            temp += mainPV.toString();
            temp += "\n";
            temp += "\n";
            std::cout << temp;
            std::cout << "Time needed: " << (getSystemTime() - endTime + time) << "\n";
        }

        if (isMateVal(local.best_score)) {
            break;
        }
    }
    return eval;
}

namespace Search {

    Depth reduce(Local &local, Board &board, Move move, bool in_pv) {
        Depth red = 0;
        const bool is_promotion = move.isPromotion(board.getPosition().K);
        if (local.depth >= 2 && !move.isCapture() && !is_promotion && local.i >= ((in_pv) ? 3 : 1)) {
            red = 1;
        }
        return red;
    }


    Value search(bool in_pv, Board &board, Line &pv, Value alpha, Value beta, Ply ply, Depth depth, Move skip_move) {
        pv.clear();
        //checking time-used


        if (ply > 0 && board.isRepetition2()) {
            return 0;
        }

        MoveListe liste;

        getMoves(board.getPosition(), liste);


        if (liste.isEmpty()) {
            return loss(ply);
        }


        if (depth <= 0) {
            return Search::qs(in_pv, board, pv, alpha, beta, ply, depth);
        }


        Local local;
        local.best_score = -INFINITE;
        local.sing_score = -INFINITE;
        local.alpha = alpha;
        local.beta = beta;
        local.ply = ply;
        local.depth = depth;
        local.move = Move{};
        local.skip_move = skip_move;


        //checking win condition


        NodeInfo info;
        Move tt_move;


        uint64_t pos_key = board.getPosition().key;


        if (!local.skip_move.isEmpty()) {
            pos_key ^= Zobrist::skip_hash;
        }

        if (ply >= MAX_PLY) {
            return board.getMover() * gameWeights.evaluate(board.getPosition(), ply);
        }




        // tb-probing
#ifndef TRAIN

        if (TT.findHash(pos_key, info)) {
            tt_move = info.tt_move;
            auto tt_score = valueFromTT(info.score, ply);
            if (info.depth >= depth && info.flag != Flag::None) {
                if ((info.flag == TT_LOWER && tt_score >= local.beta)
                    || (info.flag == TT_UPPER && tt_score <= local.alpha)
                    || info.flag == TT_EXACT) {
                    return tt_score;
                }
            }

            if ((info.flag == TT_LOWER && isWin(tt_score) && tt_score >= local.beta) ||
                (info.flag == TT_UPPER && isLoss(tt_score) && tt_score <= local.alpha)) {
                return tt_score;
            }


            if (info.flag == TT_LOWER && info.depth >= depth - 4 && isEval(tt_score) &&
                std::find(liste.begin(), liste.end(), tt_move) != liste.end()) {
                local.sing_score = tt_score;
                local.sing_move = tt_move;
            }
        }
#endif


        int start_index = 0;
        if (in_pv && ply < mainPV.length()) {
            liste.putFront(mainPV[ply]);
            start_index = 1;
        }




        //sorting
        liste.sort(board.getPosition(), ply, tt_move, start_index);

        //move-loop
        Search::move_loop(in_pv, local, board, pv, liste);


        //updating search stats
        //moved to move-loop
        if (local.best_score > local.alpha && liste.length() > 1 && board.isSilentPosition()) {
            Statistics::mPicker.update_scores(board.getPosition(), &liste.liste[0], local.move, depth);
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
        TT.storeHash(tt_value, pos_key, flag, depth, local.move);

#endif
        return local.best_score;
    }

    Value qs(bool in_pv, Board &board, Line &pv, Value alpha, Value beta, Ply ply, Depth depth) {


        pv.clear();
        nodeCounter++;
        if ((nodeCounter & 16383u) == 0u && getSystemTime() >= endTime) {
            throw std::string{"Time_out"};
        }

        if (ply >= MAX_PLY) {
            return board.getMover() * gameWeights.evaluate(board.getPosition(), ply);
        }

        if (ply > glob.sel_depth)
            glob.sel_depth = ply;


        MoveListe moves;
        getCaptures(board.getPosition(), moves);
        Value bestValue = -INFINITE;

        if (moves.isEmpty()) {
            if (board.getPosition().isEnd()) {
                return loss(ply);
            }
            if (depth == 0 && board.getPosition().hasJumps(~board.getMover())) {
                return Search::search(in_pv, board, pv, alpha, beta, ply, 1, Move{});
            }
            //bestValue = board.getMover() * gameWeights.evaluate(board.getPosition(), ply);

            if (!u_classical) {
                bestValue = network.evaluate(board.getPosition(), ply);
            } else {
                bestValue = board.getMover() * gameWeights.evaluate(board.getPosition(), ply);
            }

            if (bestValue >= beta) {
                return bestValue;
            }
        }

        if (in_pv && ply < mainPV.length()) {
            moves.putFront(mainPV[ply]);
        }
        for (int i = 0; i < moves.length(); ++i) {
            Move move = moves[i];
            Line localPV;
            board.makeMove(move);
            Value value = -Search::qs(((i == 0) ? in_pv : false), board, localPV, -beta, -std::max(alpha, bestValue),
                                      ply + 1, depth - 1);
            board.undoMove(move);
            if (value > bestValue) {
                bestValue = value;
                if (value >= beta)
                    break;
                pv.concat(move, localPV);

            }
        }

        return bestValue;
    }


    Value searchMove(bool in_pv, Move move, Local &local, Board &board, Line &line, int extension) {
        Depth reduction = Search::reduce(local, board, move, in_pv);
        //singular move extension


        if (in_pv
            && local.depth >= 8
            && move == local.sing_move
            && local.skip_move.isEmpty()
            && extension == 0
                ) {
            Value new_alpha = local.sing_score - sing_ext;
            Line new_pv;
            Value value = Search::search(in_pv, board, new_pv, new_alpha - 1, new_alpha, local.ply,
                                         local.depth - 4,
                                         move);


            if (value <= new_alpha) {
                extension = 1;
            }

        }

        Value new_alpha = std::max(local.alpha, local.best_score);

        Value val = -INFINITE;
        Depth new_depth = local.depth - 1 + extension;

        board.makeMove(move);

        if (!in_pv && new_depth > 2 && isEval(local.beta) && local.ply > 0) {

            Value margin = prob_cut;
            Value newBeta = local.beta + margin;
            Depth newDepth = std::max(new_depth - 4, 1);
            Value board_val = -qs(in_pv, board, line, -(newBeta + 1), -newBeta,
                                  local.ply + 1, newDepth);
            if (board_val >= newBeta) {
                Value value = -Search::search(in_pv, board, line, -(newBeta + 1), -newBeta, local.ply + 1,
                                              newDepth,
                                              Move{});
                if (value >= newBeta) {
                    val = value;
                }
            }
        }


        if (val == -INFINITE) {
            if ((in_pv && local.i != 0) || reduction != 0) {
                val = -Search::search(in_pv, board, line, -new_alpha - 1, -new_alpha, local.ply + 1,
                                      new_depth - reduction,
                                      Move{});
                if (val > new_alpha) {
                    val = -Search::search(in_pv, board, line, -local.beta, -new_alpha, local.ply + 1, new_depth,
                                          Move{});
                }
            } else {
                val = -Search::search(in_pv, board, line, -local.beta, -new_alpha, local.ply + 1, new_depth,
                                      Move{});
            }

        }
        board.undoMove(move);
        return val;

    }

    void move_loop(bool in_pv, Local &local, Board &board, Line &pv, MoveListe &liste) {
        local.i = 0;
        const auto num_moves = liste.length();
        int extension = (liste.length() == 1) ? 1 : 0;

        while (local.best_score < local.beta && local.i < num_moves) {
            Move move = liste[local.i];
            if (move != local.skip_move) {
                Line local_pv;
                Value value = searchMove(((local.i == 0) ? in_pv : false), move, local, board, local_pv, extension);
                if (value > local.best_score) {
                    local.move = move;
                    local.best_score = value;
                    pv.concat(move, local_pv);
                }
            }

            local.i++;
        }
        if (!local.move.isEmpty() && local.best_score >= local.beta && !local.move.isCapture()) {
            Statistics::mPicker.killer_moves[local.ply] = local.move;
        }

    }

    void search_root(Local &local, Line &line, Board &board, Value alpha, Value beta, Depth depth) {
        std::vector<Move> exluded_moves;
        return search_root(local, line, board, alpha, beta, depth, exluded_moves);
    }

    void search_root(Local &local, Line &line, Board &board, Value alpha, Value beta, Depth depth,
                     std::vector<Move> &exluded_moves) {
        line.clear();
        local.best_score = -INFINITE;
        local.sing_score = -INFINITE;
        local.alpha = alpha;
        local.beta = beta;
        local.ply = 0;
        local.depth = depth;
        local.skip_move = Move{};
        local.sing_move = Move{};
        local.move = Move{};
        MoveListe liste;
        getMoves(board.getPosition(), liste);


        //removing the excluded moves from the list

        for (Move m: exluded_moves) {
            liste.remove(m);
        }


        liste.putFront(mainPV[0]);
        int start_index = 1;

        liste.sort(board.getPosition(), local.ply, Move{}, start_index);

#ifdef TRAIN
        std::shuffle(liste.begin(), liste.end(), Zobrist::generator);
#endif


        move_loop(true, local, board, line, liste);


    }

    //there are various other things I will need to check
    void search_asp(Local &local, Board &board, Value last_score, Depth depth) {
        if (depth >= 3 && isEval(last_score)) {
            Value margin = asp_wind;
            Value alpha_margin = margin;
            Value beta_margin = margin;

            while (std::max(alpha_margin, beta_margin) < MAX_ASP) {
                Line line;
                Value alpha = last_score - alpha_margin;
                Value beta = last_score + beta_margin;
                search_root(local, line, board, alpha, beta, depth);
                Value score = local.best_score;


                if (score <= alpha) {
                    alpha_margin *= 2;
                } else if (score >= beta) {
                    beta_margin *= 2;
                } else {
                    mainPV = line;
                    return;
                }
            }
        }
        Line line;
        search_root(local, line, board, -INFINITE, INFINITE, depth);
        mainPV = line;
    }
}

Value alphaBeta(Board &board, Line &line, Ply ply, Depth depth, Value alpha, Value beta, bool in_pv) {
    nodeCounter++;
    line.clear();
    MoveListe liste;
    getMoves(board.getPosition(), liste);
    if (liste.length() == 0) {
        return loss(ply - 1);
    }
/*    alpha = std::max(alpha, loss(ply));
    if (alpha >= beta)
        return alpha;
    if (alpha >= -loss(ply))
        return alpha;*/

    int start_index = 0;
    if (in_pv && ply < mainPV.length()) {
        liste.putFront(mainPV[ply]);
        start_index = 1;
    }

    Value start_alpha = alpha;


    liste.sort(board.getPosition(), ply, Move{}, start_index);
    Move best_move;
    Depth next_depth = depth - 1;
    if (board.getPosition().hasJumps() && board.previous().hasJumps() && ply > 0) {
        next_depth++;
    } else if (next_depth == 0 &&
               board.getPosition().hasJumps(~board.getMover())) {
        next_depth++;
    }

    for (auto i = 0; i < liste.length(); ++i) {
        Move mv = liste[i];
        //Stepping through the move_list
        Value val = -INFINITE;
        board.makeMove(mv);

        Line local_pv;
        if (next_depth <= 0) {
            val = -qsSearch(board, local_pv, ply + 1, -beta, -alpha);
        } else {
            if (!in_pv && next_depth > 2 && isEval(beta) && ply >= 2) {
                //silent position here
                Value new_beta = beta + 500;
                //checking if static_eval fails
                if (-board.getMover() * gameWeights.evaluate(board.getPosition(), ply) >= new_beta) {
                    //now we can start the verification search
                    Depth verif_depth = std::max(next_depth - 4, 1);
                    Value temp = -alphaBeta(board, local_pv, ply + 1, verif_depth, -(new_beta + 1), -(new_beta),
                                            false);
                    if (temp > new_beta) {
                        val = temp;
                    }
                }
            }


            if (val == -INFINITE) {
                //Pruning goes up here
                bool do_lmr = !in_pv && !board.getPosition().hasJumps(board.getMover()) && i >= 2;
                const int pv_depth = next_depth - do_lmr;
                if (!in_pv || i > 0) {
                    val = -alphaBeta(board, local_pv, ply + 1, pv_depth, -(alpha + 1), -(alpha), false);
                    //need to glue the two pvs together
                }
                //full width search

                if (val == -INFINITE || (val > alpha && (val < beta || do_lmr))) {
                    val = -alphaBeta(board, local_pv, ply + 1, next_depth, -beta, -alpha, in_pv);
                }
            }
        }


        board.undoMove(mv);
        if (val > alpha) {
            best_move = mv;
            if (val >= beta) {
                if (!board.getPosition().hasJumps(board.getMover())) {
                    Statistics::mPicker.update_scores(board.getPosition(), &liste[0], best_move, depth);
                }
                return beta;
            }
            alpha = val;
            line.concat(best_move, local_pv);
        }
    }


    /*   if (!best_move.isEmpty() && board.isSilentPosition()) {
           Statistics::mPicker.killer_moves[ply] = best_move;
       }*/


    return alpha;
}

Value qsSearch(Board &board, Line &line, Ply ply, Value alpha, Value beta) {
    nodeCounter++;
    line.clear();
    if ((nodeCounter & 16383u) == 0u && getSystemTime() >= endTime) {
        throw std::string{"Time_out"};
    }


    MoveListe liste;
    getCaptures(board.getPosition(), liste);

    if (liste.isEmpty() || ply >= MAX_PLY) {
        return board.getMover() * gameWeights.evaluate(board.getPosition(), ply);
    }

    Move best_move;
    for (Move mv: liste) {
        board.makeMove(mv);
        Line local_pv;
        Value val = -qsSearch(board, local_pv, ply + 1, -beta, -alpha);
        board.undoMove(mv);
        if (val >= beta) {
            return beta;
        }
        if (val > alpha) {
            best_move = mv;
            alpha = val;
            line.concat(best_move, local_pv);
        }

    }


    return alpha;
}


Value search(Board board, Move &best, Depth depth, uint32_t time, bool print) {
    Statistics::mPicker.clearScores();
    glob.sel_depth = 0u;
    TT.age_counter++;
    nodeCounter = 0;
    mainPV.clear();
    //TT.clear();
    endTime = getSystemTime() + time;
    Value eval = INFINITE;
    Statistics::mPicker.clearScores();
    for (int i = 2; i <= depth; i += 2) {
        Line line;
        Value val;
        try {
            //implementing aspiration search
            val = alphaBeta(board, line, 0, i, -INFINITE, INFINITE, true);
            mainPV = line;

        } catch (std::string msg) {
            break;
        }
        best = mainPV.getFirstMove();
        eval = val;


        if (print) {
            std::string temp = std::to_string(eval) + " ";
            temp += " Depth:" + std::to_string(i) + " | " + std::to_string(glob.sel_depth) + " | ";
            temp += " NodeCount: " + std::to_string(nodeCounter) + "\n";
            temp += mainPV.toString();
            temp += "\n";
            temp += "\n";
            std::cout << temp;
        }

    }
    return eval;
}
