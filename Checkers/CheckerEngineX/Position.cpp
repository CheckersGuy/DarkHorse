#include <sstream>
#include "Position.h"
#include "Zobrist.h"

Position Position::posFromString(const std::string &pos) {
    Position result;
    for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t current = 1u << i;
        if (pos[i] == '1') {
            result.BP |= current;
        } else if (pos[i] == '2') {
            result.WP |= current;
        } else if (pos[i] == '3') {
            result.K |= current;
            result.BP |= current;
        } else if (pos[i] == '4') {
            result.K |= current;
            result.WP |= current;
        }
    }
    if (pos[32] == 'B') {
        result.color = BLACK;
    } else {
        result.color = WHITE;
    }
    result.key = Zobrist::generateKey(result);
    return result;
}

std::string Position::getPositionString()const {
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
        for (char c : token) {
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
    Scanner scanner{fen_string};
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
        if(token ==":"){
            saw_king=false;
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
    if(isEnd() || isEmpty())
        return std::string{};
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

bool Position::isEnd() const {
    return (BP == 0u && color == BLACK) || (WP == 0u && color == WHITE);
}

void Position::printPosition() const {
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
    std::cout << out.str()<< std::endl;
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
    stream.write((char *) &pos.WP, sizeof(uint32_t));
    stream.write((char *) &pos.BP, sizeof(uint32_t));
    stream.write((char *) &pos.K, sizeof(uint32_t));
    int color = (pos.color == BLACK) ? 0 : 1;
    stream.write((char *) &color, sizeof(int));
    return stream;
}

std::istream &operator>>(std::istream &stream, Position &pos) {
    stream.read((char *) &pos.WP, sizeof(uint32_t));
    stream.read((char *) &pos.BP, sizeof(uint32_t));
    stream.read((char *) &pos.K, sizeof(uint32_t));
    int color;
    stream.read((char *) &color, sizeof(int));
    pos.color = (color == 0) ? BLACK : WHITE;
    return stream;
}





