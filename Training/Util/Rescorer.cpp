
#include <iostream>
#include "SampleUtil.h"



int main(int argl, const char **argc) {

    int i, status, max_pieces, nerrors;
    EGDB_TYPE egdb_type;
    EGDB_DRIVER *handle;

    /* Check that db files are present, get db type and size. */
    status = egdb_identify(DB_PATH, &egdb_type, &max_pieces);
    std::cout<<"MAX_PIECES: "<<max_pieces<<std::endl;

    if (status) {
        printf("No database found at %s\n", DB_PATH);
        return (1);
    }
    printf("Database type %d found with max pieces %d\n", egdb_type, max_pieces);

    /* Open database for probing. */
    handle = egdb_open(EGDB_NORMAL, max_pieces, 4000, DB_PATH, print_msgs);
    if (!handle) {
        printf("Error returned from egdb_open()\n");
        return (1);
    }
    std::cout<<"Starting Rescoring the training data"<<std::endl;
    std::string in_file("../Training/TrainData/reinf.train");
    std::string out_file("../Training/TrainData/reinfformatted.train");

    create_samples_from_games(in_file, out_file, max_pieces, handle);
    std::cout<<"Done rescoring"<<std::endl;
    handle->close(handle);

    return 0;
}
