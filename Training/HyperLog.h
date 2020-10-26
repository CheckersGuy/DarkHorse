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

template<size_t num_bucket_bits, typename T, template<typename> class Hasher = std::hash>
class HyperLog {

private:
    std::array<double, (1u << num_bucket_bits)> buckets{0};
    size_t count{0};
    Hasher<T> hash;
    double alpha;
public:

    HyperLog() {
        const auto num_buckets = 1u << num_bucket_bits;
        if (num_buckets >= 128) {
            alpha = 0.7213 / (1 + 1.079 / static_cast<double>(num_buckets));
        } else if (num_buckets == 16) {
            alpha = 0.673;
        } else if (num_buckets == 32) {
            alpha = 0.697;
        } else if (num_buckets == 64) {
            alpha = 0.709;
        }


    }

    void insert(T &&value) {
        //That may work...
        using HashType = decltype(Hasher<T>::operator()(std::declval<T>()));
        constexpr auto num_bits = sizeof(HashType) * 8u;

        auto hash_value = hash(std::forward<T>(value));
        HashType maske = 0u;
        for (auto i = 0; i < num_bucket_bits; ++i) {
            maske |= 1u << (num_bits - 1 - i);
        }
        //getting the index for the bucket
        HashType bucket_index = maske & hash_value;
        bucket_index = bucket_index >> (num_bits - num_bucket_bits);

        //adding the 1...1 of the bucket index so we never have the value 0
        hash_value |= maske;


        //computing trailing zeros the old fashioned way

        size_t trailing_zeros = 0u;

        for (auto i = 0u; i < num_bits; ++i) {
            const HashType m = 1u << i;
            if ((hash_value & m) != 0) {
                trailing_zeros = i;
                break;
            }
        }


        buckets[bucket_index] = std::max(buckets[bucket_index], trailing_zeros);

        //getting the number of trailing zeros

    }

    size_t get_count() {
        double result = 0;
        const auto num_buckets = 1u << num_bucket_bits;
        double m = static_cast<double>(num_buckets);
        double eta = alpha * m * m;
        double temp = 0;

        for (auto i = 0; i < num_buckets; ++i) {
            temp += std::pow(2, -1.0 * buckets[i]);
        }
        temp = 1.0 / temp;
        eta = eta * temp;

        int num_zero_buckets = 0;
        for (auto i = 0; i < num_buckets; ++i) {
            num_zero_buckets += (buckets[i] == 0);
        }
        if (eta <= (5.0 / 2.0) * static_cast<double>(num_buckets)) {
            if (num_zero_buckets != 0) {
                //doing linear counting
                static_cast<double>(num_buckets) *
                std::log(static_cast<double>(num_buckets) / static_cast<double>(num_zero_buckets));
            } else {
                result = eta;
            }
        } else {
            result = eta;
        }

        return result;
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
