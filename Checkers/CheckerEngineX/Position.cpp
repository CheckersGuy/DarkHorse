
#include "Position.h"
#include "Bits.h"
#include "MGenerator.h"
#include "MoveListe.h"
#include "types.h"

std::optional<Move> Position::get_move(Position orig, Position next) {
  // returns the move leading from position org to next
  MoveListe liste;
  get_moves(orig, liste);
  for (auto m : liste) {
    Position t = orig;
    t.make_move(m);
    if (t == next) {
      return std::make_optional<Move>(m);
    }
  }
  return std::nullopt;
}

bool Position::has_jumps() const {
  return has_jumps(BLACK) || has_jumps(WHITE);
}

void Position::make_move(uint32_t from_index, uint32_t to_index) {
  Move move;
  move.from = 1u << from_index;
  move.to = 1u << to_index;
  make_move(move);
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
    else {
      std::cout << "INPUT: " << msg << std::endl;
      throw std::domain_error("Invalid input for fen");
    }
  }
};

Position &append_square(Position &pos, bool is_king, Color mover,
                        int num_square) {
  int index = num_square - 1;
  auto board_index = [](auto index) {
    auto row = index / 4;
    auto col = index % 4;
    return 4 * row + col;
  };
  index = board_index(index);

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
  // not correct!
  if (is_empty())
    return std::string{};
  std::ostringstream stream;
  stream << ((color == BLACK) ? "B" : "W");

  uint32_t white_pieces = WP;
  uint32_t black_pieces = BP;

  if (white_pieces) {
    stream << ":W";
  }

  auto board_index = [](auto index) {
    auto row = index / 4;
    auto col = index % 4;
    return 4 * row + col;
  };

  while (white_pieces) {
    auto square = Bits::bitscan_foward(white_pieces);

    const uint32_t mask = 1u << square;
    if (((1u << square) & K))
      stream << "K";
    square = board_index(square) + 1u;
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
    square = board_index(square) + 1;
    stream << square;
    if ((black_pieces & (~mask)) != 0u)
      stream << ",";
    black_pieces &= black_pieces - 1u;
  }

  return stream.str();
}

Position Position::get_color_flip() const {
  Position next;
  next.BP = getMirrored(WP);
  next.WP = getMirrored(BP);
  next.K = getMirrored(K);
  next.color = ~color;
  // next.key = key ^ Zobrist::color_black;
  return next;
}

bool Position::is_legal() const {
  const uint32_t b_pawns = BP & (~K);
  const uint32_t w_pawns = WP & (~K);

  const uint32_t b_kings = BP & (K);
  const uint32_t w_kings = WP & (K);

  // no pawns on the promotion squares
  if ((b_pawns & PROMO_SQUARES_BLACK) != 0)
    return false;
  if ((w_pawns & PROMO_SQUARES_WHITE) != 0)
    return false;

  // no two pieces can occupy the same square
  if ((BP & WP) != 0)
    return false;
  // ghost king
  if ((K != 0) && (K & (BP | WP)) == 0)
    return false;

  uint32_t num_wp = Bits::pop_count(WP);
  uint32_t num_bp = Bits::pop_count(BP);
  if (num_wp > 12 || num_bp > 12)
    return false;

  return true;
}

Position Position::get_start_position() {
  Position pos{};
  pos.color = BLACK;
  for (int i = 0; i <= 11; i++) {
    pos.BP |= 1u << i;
  }
  for (int i = 20; i <= 31; i++) {
    pos.WP |= 1u << i;
  }
  // pos.key = Zobrist::generate_key(pos);
  return pos;
}

bool Position::is_empty() const { return (BP == 0u && WP == 0u); }

Color Position::get_color() const { return color; }

bool Position::has_any_move() const {
  uint32_t movers =
      (color == BLACK) ? get_movers<BLACK>() : get_movers<WHITE>();
  return (movers != 0) || has_jumps(color);
}

int Position::piece_count() const { return Bits::pop_count(WP | BP); }

bool Position::has_jumps(Color color) const {
  if (color == BLACK) {
    return has_jumps<BLACK>();
  } else {
    return has_jumps<WHITE>();
  }
}

bool Position::has_threat() const { return has_jumps(~get_color()); }

bool Position::is_end() const {
  return (color == BLACK && get_movers<BLACK>() == 0u) ||
         (color == WHITE && get_movers<WHITE>() == 0u);
}

void Position::print_position() const {
  std::cout << get_pos_string() << std::endl;
}

std::string Position::get_pos_string() const {
  // here and other functions, maybe including get_fen_string is wrong
  std::string out;
  for (int row = 7; row >= 0; row--) {
    for (int col = 3; col >= 0; col--) {
      const auto bit_index = 4 * row + col;
      uint32_t maske = 1u << bit_index;

      if (row % 2 == 1) {
        out += "[ ]";
      }

      if (((BP & K) & maske) == maske) {
        out += "[B]";
      } else if (((BP)&maske) == maske) {
        out += "[0]";
      } else if (((WP & K) & maske) == maske) {
        out += "[W]";
      } else if (((WP)&maske) == maske) {
        out += "[X]";
      } else {
        out += "[ ]";
      }
      if (row % 2 == 0) {
        out += "[ ]";
      }
    }
    out += "\n";
  }

  return out;
}

void Position::make_move(Move move) {
  assert(!move.is_empty());
  // setting the piece type
  if (color == BLACK) {
    if (move.is_capture()) {
      WP &= ~move.captures;
      K &= ~move.captures;
    }
    BP &= ~move.from;
    BP |= move.to;

    if (((move.to & PROMO_SQUARES_BLACK) != 0u) && ((move.from & K) == 0))
      K |= move.to;

  } else {
    if (move.is_capture()) {
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

std::ostream &operator<<(std::ostream &stream, const Position &pos) {
  stream.write((char *)&pos, sizeof(Position));
  return stream;
}

std::istream &operator>>(std::istream &stream, Position &pos) {
  stream.read((char *)&pos, sizeof(Position));
  return stream;
}

int Position::bucket_index() { // return (piece_count() - 1) / 6;
  const auto has_kings = K != 0;
  auto pieces = piece_count();
  if (pieces == 24 || pieces == 23 || pieces == 22) {
    return 0;
  } else if (pieces == 21 || pieces == 20 || pieces == 19) {
    return 0;
  } else if (pieces == 18 || pieces == 17 || pieces == 16) {
    return 1;
  } else if (pieces == 15 || pieces == 14 || pieces == 13) {
    return 2;
  } else if (pieces == 12 || pieces == 11) {
    return 3;
  } else if (pieces == 10) {
    return 4;
  } else if (pieces == 9) {
    return 5;
  } else if (pieces == 8) {
    return 6;
  } else if (pieces == 7) {
    return 7;
  } else if (pieces == 6) {
    return 8;
  } else if (pieces == 5) {
    return 9;
  } else if (pieces == 4) {
    return 10;
  } else if (pieces == 3 || pieces == 2 || pieces == 1) {
    return 11;
  } else {
    return 0;
  }
}
