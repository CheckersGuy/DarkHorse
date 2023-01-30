#include <iostream>
#include "../generator.pb.h"
#include "../Sample.h"

std::vector<Proto::Sample> extract_sample(Proto::Game& game,int max_pieces,EGDB_DRIVER* handle);
Result get_tb_result(Proto::Sample&sample,int max_pieces,EGDB_DRIVER* handle);
