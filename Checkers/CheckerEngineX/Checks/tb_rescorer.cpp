//
// Created by leagu on 13.09.2021.
//
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "egdb.h"
#include <vector>
#include "../../Training/Sample.h"
#ifdef ITALIAN_RULES
#define DB_PATH "c:/kr_english_wld"
#else
#define DB_PATH "c:/kr_english_wld"
#endif


void print_msgs(char* msg)
{
    printf("%s", msg);
}



int main(int argl, const char **argc) {

    int i, status, max_pieces, nerrors;
    EGDB_TYPE egdb_type;
    EGDB_DRIVER *handle;

    /* Check that db files are present, get db type and size. */
    status = egdb_identify(DB_PATH, &egdb_type, &max_pieces);

    if (status) {
        printf("No database found at %s\n", DB_PATH);
        return(1);
    }
    printf("Database type %d found with max pieces %d\n", egdb_type, max_pieces);

    /* Open database for probing. */
    handle = egdb_open(EGDB_NORMAL, max_pieces, 2000, DB_PATH, print_msgs);
    if (!handle) {
        printf("Error returned from egdb_open()\n");
        return(1);
    }

    std::string path("C:/Users/leagu/Downloads/bloomcloudxx");
    std::string output("C:/Users/leagu/Downloads/bloomcloudxxrescored");
    std::ifstream stream(path, std::ios::binary);
    std::ofstream outstream(output, std::ios::binary);

    if (!stream) {
        std::cerr << "Could not find the data to be rescored" << std::endl;
    }
    std::vector<Sample> res_data;

    std::istream_iterator<Sample>begin(stream);
    std::istream_iterator<Sample>end;
    size_t total_count = 0;



  
     size_t rescored_pos = std::count_if(begin, end, [&](Sample s) {

         if (s.position.hasJumps() || Bits::pop_count(s.position.BP | s.position.WP) > max_pieces) {
             res_data.emplace_back(s);
             return false;
         }
         total_count++;
         EGDB_NORMAL_BITBOARD board;
         board.white =s.position.WP;
         board.black =s.position.BP;
         board.king =s.position.K;

         EGDB_BITBOARD normal;
         normal.normal = board;
        auto val = handle->lookup(handle, &normal, (s.position.color == BLACK) ? EGDB_BLACK : EGDB_WHITE, 0);

        if (val == EGDB_UNKNOWN) {
            std::exit(-1);
        }

        bool ret_val = (s.result == 0 && (val == EGDB_WIN || val == EGDB_LOSS) || s.result != 0 && val == EGDB_DRAW);
        if (s.result == 0 && (val == EGDB_WIN || val == EGDB_LOSS)) {
            s.result =((val == EGDB_WIN) ? s.position.color : -s.position.color);
            }
        if (s.result != 0 && val == EGDB_DRAW) {
            s.result = 0;
        }
        res_data.emplace_back(s);
            return ret_val;
        
         });
     std::cout << "Rescored positions: " << rescored_pos << "out of "<<total_count<< "possible positions"<<std::endl;

     handle->close(handle);

     std::copy(res_data.begin(), res_data.end(), std::ostream_iterator<Sample>(outstream));
    return 0;
}
