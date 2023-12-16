//
// Created by robin on 5/21/18.
//

#include "Utilities.h"
#include "MovePicker.h"

namespace Utilities {

std::unordered_set<Position> hashes;

// filling the hash to continue searching for new start_positions

void fill_hash(std::string path) {
  std::ifstream stream(path);
  std::string next_line;
  while (std::getline(stream, next_line)) {
    const auto pos = Position::pos_from_fen(next_line);
    hashes.insert(pos);
  }
}

void createNMoveBook(std::ofstream &output, int N, Board &board,
                     Value lowerBound, Value upperBound) {
  if (N == 0) {
    auto it = hashes.find(board.get_position());
    const auto pos = board.get_position();
    int material_only = 100 * Bits::pop_count(pos.WP & (~pos.K)) -
                        100 * Bits::pop_count(pos.BP & (~pos.K));
    material_only += 150 * Bits::pop_count(pos.WP & (pos.K)) -
                     150 * Bits::pop_count(pos.BP & (pos.K));

    if (it == hashes.end() && std::abs(material_only) == 0) {
      TT.clear();
      Statistics::mPicker.clear_scores();
      Board copy(board.get_position());
      Move best;
      auto value = searchValue(copy, best, 1, 1000, false, std::cout);

      if (value >= lowerBound && value <= upperBound) {
        hashes.insert(copy.get_position());
        Position currentPos = copy.get_position();
        output << currentPos.get_fen_string() << "\n";
        // std::cout << "Added position: " << hashes.size()
        //           << " with eval: " << value << "\n";
      }
    }
    return;
  }
  MoveListe liste;
  get_moves(board.get_position(), liste);
  for (int i = 0; i < liste.length(); ++i) {
    board.make_move(liste[i]);
    createNMoveBook(output, N - 1, board, lowerBound, upperBound);
    board.undo_move();
  }
}

} // namespace Utilities
