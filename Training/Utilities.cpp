#include "Utilities.h"

namespace Utils

{


std::vector<size_t> get_permutation_var1(std::vector<size_t> histogram, std::string out_path)
{
  
    std::vector<size_t> permutation;
    for (int i = 0; i < histogram.size(); ++i)
    {
        permutation.emplace_back(i);
    }

    std::sort(permutation.begin(), permutation.end(), [&](auto a, auto b)
              { return histogram[a] < histogram[b]; });
    // extending the size of the permutation to match the accumulator;
    const auto perm_half = permutation.size();
    for (auto i = 0; i < perm_half; ++i)
    {
        permutation.emplace_back(permutation[i] + perm_half);
    }

    std::ofstream out_stream(out_path, std::ios::binary);
    if (!out_stream.good())
    {
        std::cerr << "Could not load the out-stream" << std::endl;
        return permutation;
    }


    out_stream.write((char *)&permutation[0], sizeof(size_t) * permutation.size());
    std::cout << "Permutation was saved" << std::endl;
    return permutation;
}

std::vector<size_t> get_identity(std::vector<size_t> histogram){
    std::vector<size_t> identity;

    for (auto i = 0; i < histogram.size();++i){
        identity.emplace_back(i);
    }
    return identity;
}
}
