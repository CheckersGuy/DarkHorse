
#include "GameLogic.h"
#include "incbin.h"
#include "Utilities.h"
#include <filesystem>
#include "Position.h"
INCBIN(mlh_net, "mlh3.quant");
INCBIN(network, "registry_128.quant");
INCBIN(policy, "policybigger3.quant");

INCBIN(mlh_perm, "mlh.perm");
INCBIN(net_perm, "evalpermutation.perm");
INCBIN(policy_perm, "policy.perm");

int main(int argl, const char** argc){

    mlh_net.load_permutation_from_array(gmlh_permData, gmlh_permSize);
    policy.load_permutation_from_array(gpolicy_permData, gpolicy_permSize);
    network.load_permutation_from_array(gnet_permData, gnet_permSize);

    network.load_from_array(gnetworkData, gnetworkSize);
    mlh_net.load_from_array(gmlh_netData, gmlh_netSize);
    policy.load_from_array(gpolicyData, gpolicySize);

    TT.resize_in_mb(1024);
    const auto position = Position::get_start_position(); //  234
    position.print_position();

    Board board = Board(position);
    Move best;

    // searchValue(board, best, 100, 1000000, 10000000000ull, true, std::cout);

    std::cout << "Color: " << (int)position.color << std::endl;

    std::cout << "Eval" << evaluate(position, 0) << std::endl;

    const auto histogram = Utils::compute_histogramm(network, "nnzfens.fen");

    // auto perm = Utils::get_permutation_var1(histogram, "mlh.perm");
    /*
        for (int i = 0; i < histogram.size(); ++i)
        {
            std::cout << "Index: " << i << "Value: " << histogram[i] << std::endl;
        }

        auto permutation = Utils::read_vector_data<size_t>("evalpermutation.perm");

        std::cout << "Debugging: " << permutation[1] << std::endl;

        */

    // Utils::dump_evals(network, "nnzfens.fen");

    return 0;
}