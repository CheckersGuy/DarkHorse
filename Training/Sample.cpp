//
// Created by robin on 01.09.21.
//

#include "Sample.h"

std::string result_to_string(Result result) {
  if (result == BLACK_WON)
    return "BLACK_WON";
  if (result == WHITE_WON)
    return "WHITE_WON";
  if (result == DRAW)
    return "DRAW";

  return "UNKNOWN";
}
bool Sample::operator==(const Sample &other) const {
  return (position == other.position && result == other.result);
}

bool Sample::operator!=(const Sample &other) const {
  return !((*this) == other);
}

std::ofstream &operator<<(std::ofstream &stream, const Sample &s) {
  stream.write((char *)&s, sizeof(Sample));
  return stream;
}

std::ofstream &operator<<(std::ofstream &stream, Sample s) {
  stream.write((char *)&s, sizeof(Sample));
  return stream;
}

std::ifstream &operator>>(std::ifstream &stream, const Sample &s) {
  stream.read((char *)&s, sizeof(Sample));
  return stream;
}

bool Sample::is_training_sample() const {
  // excluding some positions
  auto piece_count = position.piece_count();

  return !position.has_jumps(position.color) && result != UNKNOWN &&
         (piece_count > 1) && position.has_any_move();
}
