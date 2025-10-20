#include "Utilities.h"

namespace Utils

{


std::vector<size_t> get_permutation_var1(std::vector<size_t> histogram, std::string out_path)
{
    // we are basically sorting both halves of the activations
    // according to the histogramm
    std::vector<size_t> permutation;
    for (int i = 0; i < histogram.size(); ++i)
    {
        permutation.emplace_back(i);
    }

    auto begin = permutation.begin();
    auto end = begin + (histogram.size() / 2);
    // sorting the first half
    std::sort(begin, end, [&](auto a, auto b)
              { return histogram[a] < histogram[b]; });
    // sorting the second half
    std::sort(end, permutation.end(), [&](auto a, auto b)
              { return histogram[a] > histogram[b]; });

    // saving the distribution to a file

    std::ofstream out_stream(out_path);
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
