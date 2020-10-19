#include <sstream>
#include "Position.h"
#include "Zobrist.h"


Position Position::pos_from_fen(std::string fen_string) {
    //we are assuming that fen_string is a valid pdn notation
    Position position;
    std::array<char, 4> tokens = {'B', 'b', 'W', 'w'};


    /*if (fen_string[0] == 'B')
        position.color = BLACK;
    else
        position.color = WHITE;

    std::string current;

    Color cur_color = BLACK;
    bool is_king =false;

    for (auto i = 1; i < fen_string.length(); ++i) {
        char c = fen_string[i];
        if(c==','){
            //
            is_king =false;
            current=""
        }else if(c=='')
    }
*/
    //needs to be redone
    //treat the problem as a simple state machine

    return position;
}

std::string Position::get_fen_string() const {
    std::ostringstream stream;
    stream << ((color == BLACK) ? "B" : "W");

    uint32_t white_pieces = WP;
    uint32_t black_pieces = BP;

    if (white_pieces) {
        stream << ":W";
    }

    while (white_pieces) {
        auto square = __tzcnt_u32(white_pieces);
        const uint32_t mask = 1u << square;
        if (((1u << square) & K))
            stream << "K";
        square = square + 1u;
        stream << square;
        if ((white_pieces & (~mask)) != 0u)
            stream << ",";

        white_pieces &= white_pieces - 1u;
    }

    if (black_pieces) {
        stream << ":B";
    }

    while (black_pieces) {
        auto square = __tzcnt_u32(black_pieces);
        const uint32_t mask = 1u << square;
        if (((1u << square) & K))
            stream << "K";
        square = square + 1;
        stream << square;
        if ((black_pieces & (~mask)) != 0u)
            stream << ",";
        black_pieces &= black_pieces - 1u;
    }

    return stream.str();
}

Position Position::getColorFlip() const {
    Position next;
    next.BP = getMirrored(WP);
    next.WP = getMirrored(BP);
    next.K = getMirrored(K);
    next.color = ~color;
    next.key = key ^ Zobrist::colorBlack;
    return next;
}

Position Position::getStartPosition() {
    Position pos{};
    for (int i = 0; i <= 11; i++) {
        pos.BP |= 1u << S[i];
    }
    for (int i = 20; i <= 31; i++) {
        pos.WP |= 1u << S[i];
    }
    pos.key = Zobrist::generateKey(pos);
    return pos;
}

bool Position::isEmpty() const {
    return (BP == 0u && WP == 0u);
}

Color Position::getColor() const {
    return color;
}

uint32_t Position::piece_count() {
    return __builtin_popcount(WP | BP);
}

bool Position::hasJumps(Color col) const {
    if (col == BLACK) {
        return getJumpers<BLACK>() != 0u;
    } else {
        return getJumpers<WHITE>() != 0u;
    }
}

bool Position::hasThreat() const {
    return hasJumps(~getColor());
}

bool Position::isWipe() const {
    return ((getColor() == BLACK && getMovers<BLACK>() == 0u) || (getColor() == WHITE && getMovers<WHITE>() == 0u));
}

std::string Position::position_str() const {
    std::ostringstream out;
    uint32_t counter = 32u;
    for (int i = 0; i < 64; i++) {
        int row = i / 8;
        int col = i % 8;
        if ((row + col) % 2 == 0) {
            out << "[ ]";
        } else {
            if ((row + col + 1) % 2 == 0) {
                counter--;
            }
            uint32_t maske = 1u << (counter);
            if (((BP & K) & maske) == maske) {
                out << "[B]";
            } else if (((BP) & maske) == maske) {
                out << "[0]";
            } else if (((WP & K) & maske) == maske) {
                out << "[W]";
            } else if (((WP) & maske) == maske) {
                out << "[X]";
            } else {
                out << "[ ]";
            }
        }
        if ((i + 1) % 8 == 0) {
            out << "\n";
        }
    }
    return out.str();
}

void Position::printPosition() const {
    std::cout << position_str() << std::endl;
}

void Position::makeMove(Move &move) {
    assert(!move.isEmpty());
    //setting the piece type
    if (color == BLACK) {
        if (move.isCapture()) {
            WP &= ~move.captures;
            K &= ~move.captures;
        }
        BP &= ~move.from;
        BP |= move.to;

        if (((move.to & PROMO_SQUARES_BLACK) != 0u) && ((move.from & K) == 0))
            K |= move.to;

    } else {
        if (move.isCapture()) {
            BP &= ~move.captures;
            K &= ~move.captures;
        }
        WP &= ~move.from;
        WP |= move.to;

        if (((move.to & PROMO_SQUARES_WHITE) != 0u) && ((move.from & K) == 0))
            K |= move.to;

    }
    if ((move.from & K) != 0) {
        K &= ~move.from;
        K |= move.to;
    }
    this->color = ~this->color;
}
