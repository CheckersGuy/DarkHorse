
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


    /*
    const auto histogram = Utils::compute_histogramm(network, "nnzfens.fen");
    auto permutation = Utils::get_identity(histogram);
    Utils::get_permutation_var1(histogram,"evalpermutation.perm");

    for (int i = 0;i< histogram.size();++i)
    {
        std::cout << "Index: " << i << "Value: " << histogram[i] << std::endl;
    }
    */

    // loading a permutation from file;

    std::filesystem::path my_path("nnzfens.fen");


    const auto file_size = std::filesystem::file_size(my_path);
    std::cout << "FileSize: " <<  file_size << std::endl;
    const auto perm_size = file_size / sizeof(size_t);
    std::vector<size_t> permutation(perm_size);

    std::ifstream in_stream("nnzfens.fen", std::ios::binary);

    if (!in_stream.good()) {
            std::cerr << "Error could not load the stream" << std::endl;
            return 1;
    }

    in_stream.read((char *)permutation[0], file_size);

    for (auto val : permutation){
        std::cout << val << std::endl;
    }

        return 0;
}