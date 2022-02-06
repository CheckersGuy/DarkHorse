#include "GameLogic.h"


Line mainPV;
uint64_t endTime = 1000000000;
uint64_t nodeCounter = 0u;


Weights<int16_t> gameWeights;


SearchGlobal glob;
Network network,network2;
bool u_classical = false;
Value last_eval;
void initialize() {
    gameWeights.load_weights<uint32_t>("../Training/Engines/sgd.weights");
    Zobrist::init_zobrist_keys();

}

void initialize(uint64_t seed) {
    gameWeights.load_weights<uint32_t>("../Training/Engines/small.weights");
    Zobrist::init_zobrist_keys(seed);

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

    //if there is only one move we can return

    MoveListe liste;
    get_moves(board.get_position(), liste);



    endTime = getSystemTime() + time;
    auto start_time =getSystemTime();
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
            std::cout << "Time needed: " << (getSystemTime() - start_time) << "\n";
        }

        if (isMateVal(local.best_score)) {
            break;
        }
    }
    last_eval = eval;
    return eval;
}

namespace Search {

    Depth reduce(Local &local, Board &board, Move move, bool in_pv) {
        Depth red = 0;
        const bool is_promotion = move.is_promotion(board.get_position().K);
        if (local.depth >= 2 && !move.is_capture() && !is_promotion && local.i >= ((in_pv) ? 3 : 1)) {
            red = 1;
        }
        return red;
    }


    Value search(bool in_pv, Board &board, Line &pv, Value alpha, Value beta, Ply ply, Depth depth, Move skip_move) {
        pv.clear();
        nodeCounter++;
        //checking time-used
        if (ply > 0 && board.isRepetition2()) {
            return 0;
        }

        MoveListe liste;

        get_moves(board.get_position(), liste);


        if (liste.is_empty()) {
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


        uint64_t pos_key = board.get_position().key;


        if (!local.skip_move.is_empty()) {
            pos_key ^= Zobrist::skip_hash;
        }

        if (ply >= MAX_PLY) {
            return board.get_mover() * gameWeights.evaluate(board.get_position(), ply);
        }




        // tb-probing
#ifndef TRAIN

        if (TT.find_hash(pos_key, info)) {
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
            liste.put_front(mainPV[ply]);
            start_index = 1;
        }




        //sorting
        liste.sort(board.get_position(), depth, ply, tt_move, start_index);

        //move-loop
        Search::move_loop(in_pv, local, board, pv, liste);


        //updating search stats
        //moved to move-loop
        if (local.best_score > local.alpha && liste.length() > 1 && board.is_silent_position()) {
            Statistics::mPicker.update_scores(board.get_position(), &liste.liste[0], local.move, depth);
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
        TT.store_hash(tt_value, pos_key, flag, depth, local.move);

#endif
        return local.best_score;
    }

    Value qs(bool in_pv, Board &board, Line &pv, Value alpha, Value beta, Ply ply, Depth depth) {
        pv.clear();
        nodeCounter++;
        if ((nodeCounter & 8383u) == 0u && getSystemTime() >= endTime) {
            throw std::string{"Time_out"};
        }

        if (ply >= MAX_PLY) {
            return board.get_mover() * gameWeights.evaluate(board.get_position(), ply);
        }

        if (ply > glob.sel_depth)
            glob.sel_depth = ply;


        MoveListe moves;
        get_captures(board.get_position(), moves);
        Value bestValue = -INFINITE;

        if (moves.is_empty()) {
            if (board.get_position().is_end()) {
                return loss(ply);
            }
            if (depth == 0 && board.get_position().has_jumps(~board.get_mover())) {
                return Search::search(in_pv, board, pv, alpha, beta, ply, 1, Move{});
            }
            //bestValue = board.get_mover() * gameWeights.evaluate(board.get_position(), ply);

            if (!u_classical) {
                bestValue = Network::evaluate(network, network2, board.get_position(), ply);
            } else {
                bestValue = board.get_mover() * gameWeights.evaluate(board.get_position(), ply);
            }

            if (bestValue >= beta) {
                return bestValue;
            }
        }

        if (in_pv && ply < mainPV.length()) {
            moves.put_front(mainPV[ply]);
        }
        for (int i = 0; i < moves.length(); ++i) {
            Move move = moves[i];
            Line localPV;
            board.make_move(move);
            Value value = -Search::qs(((i == 0) ? in_pv : false), board, localPV, -beta, -std::max(alpha, bestValue),
                                      ply + 1, depth - 1);
            board.undo_move(move);
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
            && local.skip_move.is_empty()
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

        board.make_move(move);

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
        board.undo_move(move);
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
        get_moves(board.get_position(), liste);


        //removing the excluded moves from the list

        for (Move m: exluded_moves) {
            liste.remove(m);
        }


        liste.put_front(mainPV[0]);
        int start_index = 1;

        liste.sort(board.get_position(), local.depth, local.ply, Move{}, start_index);

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
