//
// Created by root on 19.05.21.
//

#include "GeneratorZ.h"

std::ostream &operator<<(std::ostream &stream, const TrainSample &s) {

    stream << s.pos;
    stream.write((char *) &s.result, sizeof(int));
    stream.write((char *) &s.evaluation, sizeof(int));
    return stream;
}

std::istream &operator>>(std::istream &stream, TrainSample &s) {
    stream >> s.pos;
    int result;
    stream.read((char *) &result, sizeof(int));
    s.result = result;
    int eval;
    stream.read((char *) &eval, sizeof(int));
    s.evaluation = eval;
    return stream;
}

void GeneratorZ::generate_games() {
    //we are generating games


    Position start;
    constexpr size_t MAX_MOVES = 500;

    Board board;
    board = start;

    int result = 0;
    while (num_games < max_games) {
        std::vector<TrainSample> local_samples;
        for (auto i = 0; i < MAX_MOVES; ++i) {
            MoveListe liste;
            getMoves(board.getPosition(), liste);
            if (liste.isEmpty()) {
                //check
                result = (board.getMover() == BLACK) ? 1 : -1;
                break;
            }

            //finish that implementation

            TrainSample sample;
            Move best;
            auto eval = searchValue(board, best, (int) max_depth, max_time, false);
            sample.pos = board.getPosition();
            sample.evaluation = board.getMover() * eval;
            local_samples.emplace_back(sample);
            board.makeMove(best);
        }
        for (TrainSample &s : local_samples) {
            s.result = result;
        }
    }

}