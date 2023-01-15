#include <iostream>
#include <fstream>
#include "Position.h"
#include "GameLogic.h"
#include <unordered_set>
//will be used to generate opening positions for training
namespace Book{
void create_train_file(std::string base_book, std::string output, int depth);

void recursive_collect(Board& board,int depth,std::unordered_set<Position>&set,std::ofstream& out_stream);
}
