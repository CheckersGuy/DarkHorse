#include "Network.h"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>

static constexpr int BLOCK_ROWS = 2;
static constexpr int BLOCK_COLS = 2;
static constexpr int INPUT_SIZE = 8; // number of columns in the input matrix
/*
int get_weight_index(int index) {
  const int ROW = index / INPUT_SIZE;
  const int COL = index % INPUT_SIZE;
  const int BLOCK_SIZE = BLOCK_ROWS * BLOCK_COLS;
  int NUM_ROW_BLOCKS = INPUT_SIZE / BLOCK_COLS;

  int row = ROW / BLOCK_COLS;
  int col = COL / BLOCK_COLS;
  //  std::cout << "RowBlock: " << row << std::endl;
  //  std::cout << "RowCol:" << col << std::endl;
  int block_row = row % BLOCK_ROWS;
  int block_col = row % BLOCK_COLS;

  // std::cout << "BlockRow: " << block_row << std::endl;
  // std::cout << "BlockCol: " << block_col << std::endl;
  // std::cout << "Index: (" << row << ", " << col << ")" << std::endl;

  return (INPUT_SIZE / BLOCK_COLS) * row * BLOCK_SIZE + col * BLOCK_SIZE +
         block_row * BLOCK_ROWS + block_col;
}

int main(const int argl, const char **argc) {

  const int rows = 8;
  const int cols = 8;

  int8_t array[rows * cols] = {0};
  int8_t layout[rows * cols] = {0};

  std::mt19937_64 generator;
  std::uniform_int_distribution<int8_t> distrib(-9, 9);

  for (auto i = 0; i < rows; ++i) {
    for (auto j = 0; j < cols; ++j) {
      array[i] = distrib(generator);
      std::cout << (int)array[i] << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "Test" << std::endl;
  int i = 4;
  int j = 2;
  int index = i * INPUT_SIZE + j;
  auto test_index = get_weight_index(index);
  std::cout << "Index: " << test_index << std::endl;
}
*/
