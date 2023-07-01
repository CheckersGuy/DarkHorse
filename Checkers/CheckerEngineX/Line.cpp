//
// Created by robin on 9/17/18.
//

#include "Line.h"

Line::Line(const Line &other) {

  for (auto i = 0; i < other.counter; ++i) {
    myArray[i] = other.myArray[i];
  }
  counter = other.counter;
}

int Line::length() const { return counter; }

void Line::addMove(const Move &move) { this->myArray[counter++] = move; }

void Line::concat(const Move &best, const Line &line) {
  myArray[0] = best;
  for (int i = 0; i < line.length(); ++i) {
    myArray[i + 1] = line.myArray[i];
  }
  this->counter = line.counter + 1;
}

std::string Line::toString() const {
  std::string current;
  for (int k = 0; k < length(); ++k) {
    current += std::to_string(myArray[k].get_from_index() + 1) + "-";
    current += std::to_string(myArray[k].get_to_index() + 1) + " ";
  }
  return current;
}

std::ostream &operator<<(std::ostream &stream, Line &line) {
  stream << line.toString();
  return stream;
}

Move &Line::getFirstMove() { return myArray[0]; }

void Line::clear() { this->counter = 0; }

bool Line::operator==(const Line &other) const {
  if (other.length() != this->length())
    return false;

  for (int i = 0; i < other.length(); ++i) {
    if (myArray[i] != other.myArray[i])
      return false;
  }

  return true;
}

bool Line::operator!=(const Line &other) const { return !(*this == other); }

const Move &Line::operator[](int index) const { return this->myArray[index]; }

Move &Line::operator[](int index) { return this->myArray[index]; }

Line &Line::operator=(const Line &other) {
  for (auto i = 0; i < other.counter; ++i) {
    myArray[i] = other.myArray[i];
  }
  counter = other.counter;

  return *this;
}
