#include "Sample.h"
#include "egdb.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#define DB_PATH "D:\\kr_english_wld"

void print_msgs(char *msg) { printf("%s", msg); }

enum class SampleResult : int { DRAW = 1, WIN = 2, LOSS = 3, UNKNOWN = 0 };
struct SampleData {
  std::string fen_string;
  Position pos;
  int16_t eval;
  int8_t result;

  friend std::ifstream &operator>>(std::ifstream &stream, SampleData &other) {
    uint16_t size;
    stream.read((char *)&size, sizeof(uint16_t));
    // std::cout << "Size: " << (int)size << std::endl;
    other.fen_string.resize(size);
    stream.read((char *)other.fen_string.c_str(), sizeof(char) * size);
    stream.read((char *)&other.eval, sizeof(int16_t));
    stream.read((char *)&other.result, sizeof(int8_t));
    return stream;
  }
  friend std::ofstream &operator<<(std::ofstream &stream, SampleData other) {
    uint16_t size = other.fen_string.size();
    // std::cout << "Writing: " << (int)size << std::endl;
    // std::cout << "TheStringIs: " << other.fen_string << std::endl;
    stream.write((char *)&size, sizeof(uint16_t));
    stream.write((char *)other.fen_string.c_str(), sizeof(char) * size);
    stream.write((char *)&other.eval, sizeof(int16_t));
    stream.write((char *)&other.result, sizeof(int8_t));
    return stream;
  }
};

Result get_tb_result(Position pos, int max_pieces, EGDB_DRIVER *handle) {
  if (pos.has_jumps() || Bits::pop_count(pos.BP | pos.WP) > max_pieces)
    return UNKNOWN;

  EGDB_NORMAL_BITBOARD board;
  board.white = pos.WP;
  board.black = pos.BP;
  board.king = pos.K;

  EGDB_BITBOARD normal;
  normal.normal = board;
  auto val = handle->lookup(
      handle, &normal, ((pos.color == BLACK) ? EGDB_BLACK : EGDB_WHITE), 0);

  if (val == EGDB_UNKNOWN)
    return UNKNOWN;

  if (val == EGDB_WIN)
    return (pos.color == BLACK) ? BLACK_WON : WHITE_WON;

  if (val == EGDB_LOSS)
    return (pos.color == BLACK) ? WHITE_WON : BLACK_WON;

  if (val == EGDB_DRAW)
    return DRAW;

  return UNKNOWN;
}
// rescores data and outputs a new data format
void rescore_data(int max_pieces, std::string input, std::string output) {
  // to be continued
}

int main(int argl, const char **argc) {

  int i, status, max_pieces, nerrors;
  EGDB_TYPE egdb_type;
  EGDB_DRIVER *handle;

  /* Check that db files are present, get db type and size. */
  status = egdb_identify(DB_PATH, &egdb_type, &max_pieces);
  std::cout << "MAX_PIECES: " << max_pieces << std::endl;

  if (status) {
    printf("No database found at %s\n", DB_PATH);
    return (1);
  }
  printf("Database type %d found with max pieces %d\n", egdb_type, max_pieces);

  /* Open database for probing. */
  handle = egdb_open(EGDB_NORMAL, max_pieces, 2000, DB_PATH, print_msgs);
  if (!handle) {
    printf("Error returned from egdb_open()\n");
    return (1);
  }

  std::cout << "Hello I am the rescorer" << std::endl;
  std::string file_name = argc[1];
  std::string path = "../Training/TrainData/" + file_name;
  std::cout << "Path: " << path << std::endl;
  std::ifstream stream(path.c_str(), std::ios::binary);
  if (!stream.good()) {
    std::cout << "Could not open the input_stream" << std::endl;
  }
  std::ofstream out_stream((path + ".rescored").c_str(), std::ios::binary);
  if (!out_stream.good()) {
    std::cout << "Could not open the output_stream" << std::endl;
  }

  Sample test;
  int total_counter = 0;
  int wrong_counter = 0;
  while (stream >> test) {
    // std::cout << test << std::endl;
    total_counter++;
    if (test.position.get_color() == BLACK) {
      test.position = test.position.get_color_flip();
      if (test.result != DRAW && test.result != UNKNOWN) {
        test.result = (test.result == WHITE_WON) ? BLACK_WON : WHITE_WON;
      }
    }
    auto result = get_tb_result(test.position, 10, handle);

    // newDataFormat
    SampleData new_format;
    new_format.fen_string = test.position.get_fen_string();
    new_format.result = test.result;
    out_stream << new_format;

    if (result != UNKNOWN && result != test.result) {
      wrong_counter++;
      if ((wrong_counter + 1) % 10000) {
        std::cout << wrong_counter << std::endl;
      }
      /*
          auto result_string = (result == WHITE_WON) ? "WHITE_WON"
                               : (result == DRAW)    ? "DRAW"
                                                     : "BLACK_WON";

          std::cout << test << "\n";
          std::cout << "TableBaseResult : " << result_string << std::endl;

          std::cout << "----------------------------" << std::endl;
          */
    }
    if (result != UNKNOWN) {
      test.result = result;
    }
  }

  handle->close(handle);
  std::cout << "TotalCounter: " << total_counter << std::endl;
  std::cout << "WrongCounter: " << wrong_counter << std::endl;
  /*
  {
    SampleData test;
    test.fen_string = "B:WK29:BK4";
    std::ofstream out_stream("test.data", std::ios::binary);
    if (!out_stream.good()) {
      std::exit(-1);
    }
    out_stream << test;
  }
  {
    std::ifstream in_stream("test.data", std::ios::binary);
    if (!in_stream.good()) {
      std::exit(-1);
    }
    SampleData other;
    in_stream >> other;
    std::cout << "OtherFenString: " << other.fen_string << std::endl;
  }
  */
  return 0;
}
/*
int main(int argl, const char **argc) {
  std::cout << "Hello I am the rescorer" << std::endl;

  std::string file_name = argc[1];
  std::string path = "../Training/TrainData/" + file_name;
  std::cout << "Path: " << path << std::endl;
  std::ifstream stream(path.c_str());
  if (!stream.good()) {
    std::cout << "Could not open the stream" << std::endl;
  }

  Sample test;

  while (stream >> test) {
    std::cout << test << std::endl;
    std::cout << "----------------------------" << std::endl;
  }
}
*/
