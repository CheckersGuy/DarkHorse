//
// Created by root on 25.10.20.
//

#ifndef READING_HYPERLOG_H
#define READING_HYPERLOG_H

#include <cstdint>
#include <cstddef>
#include <array>
#include <cmath>
#include "immintrin.h"
#include <bitset>

//default is around 8kb of memory
template< typename T,size_t num_bucket_bits =10, typename Hasher = std::hash<T>>
class HyperLog {

private:
    std::array<double, (1u << num_bucket_bits)> buckets{0};
    size_t count{0};
    Hasher hash;
    double alpha;

    template<typename S>
    void print_bit_string(S value) {
        static_assert(std::is_unsigned_v<S>);
        for (uint32_t i = 0; i < sizeof(S) * 8; ++i) {
            const S maske = S{1u} << (sizeof(S) * 8 - i - 1);
            if ((maske & value) != 0u)
                std::cout << "1";
            else
                std::cout << "0";

        }
        std::cout << std::endl;

    }

public:

    HyperLog() {
        const auto num_buckets = 1u << num_bucket_bits;
        if (num_buckets >= 128) {
            alpha = 0.7213 / (1 + 1.079 / ((double) num_buckets));
        } else if (num_buckets == 16) {
            alpha = 0.673;
        } else if (num_buckets == 32) {
            alpha = 0.697;
        } else if (num_buckets == 64) {
            alpha = 0.709;
        }

    }

    void insert(T value) {
        //That may work...
        using HashType = decltype(hash.operator()(std::declval<T>()));
        static_assert(std::is_unsigned_v<HashType>);
        constexpr auto num_bits = sizeof(HashType) * 8u;

        auto hash_value = hash(std::forward<T>(value));
        HashType maske{0u};

        for (uint32_t i = 0; i < num_bucket_bits; ++i) {
            maske |= HashType{1u} << (num_bits - i - 1);
        }


        //getting the index for the bucket
        HashType bucket_index = maske & hash_value;
        bucket_index = bucket_index >> (num_bits - num_bucket_bits);

        //adding the 1...1 of the bucket index so we never have the value 0
        hash_value |= maske;


        //computing trailing zeros the old fashioned way

        size_t trailing_zeros = 1u;

        for (auto i = (num_bits-num_bucket_bits-1); i >=0; --i) {
            const HashType m = HashType{1u} << i;
            if ((hash_value & m) == 0) {
                trailing_zeros ++;
            }else{
                break;
            }
        }
        buckets[bucket_index] = std::max(buckets[bucket_index], (double) trailing_zeros);

        //getting the number of trailing zeros

    }

    size_t get_count() {
        double result;
        const auto num_buckets = 1u << num_bucket_bits;
        double m = num_buckets;
        double temp = 0;
        for (auto i = 0; i < num_buckets; ++i) {
            temp += 1.0/std::pow(2.0,buckets[i]);
        }
        temp = 1.0 / temp;


        double eta = alpha * m * m * temp;

        int num_zero_buckets = 0;
        for (auto i = 0; i < num_buckets; ++i) {
            if(buckets[i]==0)
                num_zero_buckets++;
        }
        std::cout<<num_zero_buckets<<std::endl;
        if (eta <= (5.0 / 2.0) * ((double) num_buckets)) {
            if (num_zero_buckets != 0) {
                result = ((double) num_buckets) *
                         std::log(((double) num_buckets) / ((double) num_zero_buckets));
            } else {
                result = eta;
            }
        } else if(eta<=(1.0/30.0)*std::pow(2.0,32.0)){
            result = eta;
        }else{
            result = -std::pow(2.0,32.0)*std::log(1-eta/std::pow(2.0,32.0));
        }

        return std::round(result);
    }

    HyperLog &operator+=(const HyperLog &other) {
        //merging the buckets
        for (auto i = 0; i < buckets.size(); ++i) {
            buckets[i] += other.buckets[i];
        }
        return *this;
    }


};


#endif //READING_HYPERLOG_H
