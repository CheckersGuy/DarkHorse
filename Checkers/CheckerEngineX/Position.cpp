#include <sstream>
#include "Zobrist.h"

#include "Position.h"

bool Position::hasJumps() const {
    return hasJumps(BLACK) || hasJumps(WHITE);
}

std::string Position::getPositionString() const {
    std::string position;
    for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t current = 1u << i;
        if ((current & (BP & K))) {
            position += "3";
        } else if ((current & (WP & K))) {
            position += "4";
        } else if ((current & BP)) {
            position += "1";
        } else if ((current & WP)) {
            position += "2";
        } else {
            position += "0";
        }
    }
    if (getColor() == BLACK) {
        position += "B";
    } else {
        position += "W";
    }
    return position;
}


struct Scanner {
    std::string msg;
    int index{0};

    std::string get_token() {
        if (msg.empty())
            return "";
        if (index >= msg.size())
            return "";

        std::string token = "";
        char c = msg[index];
        if (isdigit(c) && index < msg.size() - 1 && isdigit(msg[index + 1])) {
            token += c;
            token += msg[index + 1];
            index += 2;
            return token;
        } else if (isdigit(c)) {
            token += c;
            index += 1;
            return token;
        }

        if (c == ':' || c == ',' || c == 'B' || c == 'W' || c == 'K') {
            token += c;
            index++;
            return token;
        }
        throw std::domain_error("Invalid input for fen");

    }

    bool is_square(std::string token) {
        for (char c: token) {
            if (!isdigit(c))
                return false;
        }
        int num_square = std::stoi(token);
        if (1 <= num_square && num_square <= 32)
            return true;
        else
            throw std::domain_error("Invalid input for fen");
    }

};

Position &append_square(Position &pos, bool is_king, Color mover, int num_square) {
    int index = num_square - 1;

    if (mover == BLACK) {
        pos.BP |= 1u << index;
    } else if (mover == WHITE) {
        pos.WP |= 1u << index;
    }
    if (is_king)
        pos.K |= 1u << index;
    return pos;
}


Position Position::pos_from_fen(std::string fen_string) {

    Position position;
    Scanner scanner{std::move(fen_string)};
    bool saw_king = false;

    auto first_token = scanner.get_token();
    if (first_token == "B")
        position.color = BLACK;
    else
        position.color = WHITE;

    Color mover;

    while (true) {
        auto token = scanner.get_token();
        if (token.empty())
            break;

        if (token == "K") {
            saw_king = true;
        }
        if (token == "B") {
            mover = BLACK;
        }
        if (token == "W") {
            mover = WHITE;
        }
        if (token == ":") {
            saw_king = false;
        }

        if (token == ",") {
            saw_king = false;
        }
        if (scanner.is_square(token)) {
            int num_square = std::stoi(token);
            append_square(position, saw_king, mover, num_square);
        }
    }

    return position;
}

std::string Position::get_fen_string() const {
    if (isEmpty())
        return std::string{};
    std::ostringstream stream;
    stream << ((color == BLACK) ? "B" : "W");

    uint32_t white_pieces = WP;
    uint32_t black_pieces = BP;

    if (white_pieces) {
        stream << ":W";
    }

    while (white_pieces) {
        auto square = Bits::bitscan_foward(white_pieces);
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
        auto square = Bits::bitscan_foward(black_pieces);
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


PieceType Position::getPieceType(Move move) const {
    PieceType type;
    if ((move.from & (BP & K)) != 0) {
        type = BKING;
    } else if ((move.from & (WP & K)) != 0) {
        type = WKING;
    } else if ((move.from & BP) != 0) {
        type = BPAWN;
    } else if ((move.from & WP) != 0) {
        type = WPAWN;
    } else {
        type = WPAWN;
    }
    return type;
}

bool Position::islegal() const {
    const uint32_t b_pawns = BP & (~K);
    const uint32_t w_pawns = WP & (~K);

    const uint32_t b_kings = BP & (K);
    const uint32_t w_kings = WP & (K);

    //no pawns on the promotion squares
    if ((b_pawns & PROMO_SQUARES_BLACK) != 0)
        return false;
    if ((w_pawns & PROMO_SQUARES_WHITE) != 0)
        return false;

    //no two pieces can occupy the same square
    if ((BP & WP) != 0)
        return false;
    //ghost king
    if ((K != 0) && (K & (BP | WP)) == 0)
        return false;

    return true;
}

Position Position::getStartPosition() {
    Position pos{};
    pos.color = BLACK;
    for (int i = 0; i <= 11; i++) {
        pos.BP |= 1u << i;
    }
    for (int i = 20; i <= 31; i++) {
        pos.WP |= 1u << i;
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
    return Bits::pop_count(WP | BP);
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

bool Position::isEnd() const {
    return (color == BLACK && getMovers<BLACK>() == 0u) || (color == WHITE && getMovers<WHITE>() == 0u);
}

void Position::printPosition() const {
    std::string out;
    uint32_t counter = 32u;
    for (int i = 0; i < 64; i++) {
        int row = i / 8;
        int col = i % 8;
        if ((row + col) % 2 == 0) {
            out += "[ ]";
        } else {
            if ((row + col + 1) % 2 == 0) {
                counter--;
            }
            uint32_t maske = 1u << (counter);
            if (((BP & K) & maske) == maske) {
                out += "[B]";
            } else if (((BP) & maske) == maske) {
                out += "[0]";
            } else if (((WP & K) & maske) == maske) {
                out += "[W]";
            } else if (((WP) & maske) == maske) {
                out += "[X]";
            } else {
                out += "[ ]";
            }
        }
        if ((i + 1) % 8 == 0) {
            out += "\n";
        }
    }
    std::cout << out << std::endl;
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

uint32_t Position::getKingAttackSquares(uint32_t bit_mask) {
    uint32_t squares = defaultShift<BLACK>(bit_mask) | forwardMask<BLACK>(bit_mask);
    squares |= defaultShift<WHITE>(bit_mask) | forwardMask<WHITE>(bit_mask);
    squares &= ~(BP | WP);
    return squares;
}

std::ostream &operator<<(std::ostream &stream, const Position &pos) {
    stream.write((char *) &pos, sizeof(Position));
    return stream;
}

std::istream &operator>>(std::istream &stream, Position &pos) {
    stream.read((char *) &pos, sizeof(Position));
    return stream;
}

std::ostream &operator<<(std::ostream &stream, Square square) {

    return stream;
}





