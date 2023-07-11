#include "Book.h"

namespace Book {

size_t pos_counter = 0;
size_t op_index = 0;

void create_train_file(std::string base_book, std::string output, int depth) {
  pos_counter = 0;
  op_index = 0;
  TT.resize(20);
  network.load_bucket("testing6.quant");
  std::unordered_set<Position> positions;
  std::ofstream out_stream(output, std::ios::app);
  std::ifstream in_stream(base_book);
  if (!out_stream.good()) {
    std::cerr << "Could not load the output streams" << std::endl;
    std::exit(-1);
  }
  if (!in_stream.good()) {
    std::cerr << "Could not load the input streams" << std::endl;
    std::exit(-1);
  }
  std::istream_iterator<Position> begin(in_stream);
  std::istream_iterator<Position> end;
  /*
    std::for_each(begin, end, [&](Position pos) {
      Board board;
      board = pos;
      op_index++;
      recursive_collect(board, depth, positions, out_stream);
    });
  */
  Board board;
  board = Position::get_start_position();
  recursive_collect(board, depth, positions, out_stream);

  std::cout << "Collected: " << positions.size() << std::endl;
}

void recursive_collect(Board &board, int depth,
                       std::unordered_set<Position> &set,
                       std::ofstream &out_stream) {
  if (depth <= 0) {
    // checking if we should add the position
    if (set.find(board.get_position()) == set.end()) {
      set.insert(board.get_position());

      Position current = board.get_position();
      Move best;
      Board copy;
      copy = board.get_position();
      auto value = searchValue(copy, best, 3, 10, false, std::cout);

      if (std::abs(value) <= 130 && !copy.get_position().has_jumps() &&
          copy.get_position().piece_count() >= 20) {
        std::cout << board.get_mover() * value << std::endl;
        std::cout << "Added a position" << std::endl;
        std::cout << "Index: " << op_index << std::endl;
        std::cout << "Count: " << pos_counter++ << std::endl;
        current.print_position();
        std::cout << "\n\n";
        out_stream << current.get_fen_string() << "\n";
      }
    }

    return;
  }

  MoveListe liste;
  get_moves(board.get_position(), liste);

  for (auto m : liste) {
    board.make_move(m);
    recursive_collect(board, depth - 1, set, out_stream);
    board.undo_move();
  }
}
} // namespace Book
