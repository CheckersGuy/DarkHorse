//
// Created by robin on 18.12.21.
//


#include "File.h"

namespace File {


    size_t num_illegal_samples(std::string input){
        std::ifstream stream(input,std::ios::binary);
        std::istream_iterator<Sample>begin(stream);
        std::istream_iterator<Sample>end;

        return std::count_if(begin,end,[](Sample s){return !s.position.islegal();});
    }

    void remove_duplicates(std::string input, std::string output) {
        std::ifstream stream(input, std::ios::binary);
        std::istream_iterator<Sample> begin(stream);
        std::istream_iterator<Sample> end;

        size_t counter = 0;
        size_t total_elements = 0;
        SampleFilter filter(5751035027, 10);
        std::vector<Sample> elements;
        std::for_each(begin, end, [&](Sample s) {
            if (!filter.has(s)) {
                counter++;
                filter.insert(s);
                elements.emplace_back(s);
            }
            total_elements++;
        });

        std::ofstream out_stream(output, std::ios::binary);
        std::copy(elements.begin(), elements.end(), std::ostream_iterator<Sample>(out_stream));
        std::cout << "Size after removing: " << elements.size() << std::endl;
        std::cout << "Removed a total of " << total_elements - elements.size() << " elements" << std::endl;
    }

}