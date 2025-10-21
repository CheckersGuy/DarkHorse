
#include "GameLogic.h"
#include "incbin.h"
#include "Utilities.h"
#include <filesystem>
INCBIN(mlh_net, "mlh3.quant");
INCBIN(network, "registry_128.quant");
INCBIN(policy, "policybigger3.quant");



int main(int argl, const char** argc){

    mlh_net.load_from_array(gmlh_netData, gmlh_netSize);
    network.load_from_array(gnetworkData, gnetworkSize);
    policy.load_from_array(gpolicyData, gpolicySize);

    const auto histogram = Utils::compute_histogramm(network, "nnzfens.fen");
    auto perm = Utils::get_permutation_var1(histogram, "evalpermutation.perm");

    for (int i = 0; i < histogram.size(); ++i)
    {
        std::cout << "Index: " << i << "Value: " << histogram[i] << std::endl;
    }

    auto permutation = Utils::read_vector_data<size_t>("evalpermutation.perm");

    std::cout << "Debugging: " << permutation[1] << std::endl;

    return 0;
}