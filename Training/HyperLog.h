//
// Created by root on 25.10.20.
//

#ifndef READING_HYPERLOG_H
#define READING_HYPERLOG_H

#include <cstddef>
#include <array>
#include <cmath>
#include "immintrin.h"

//default is around 8kb of memory
//Notes:

/*the maximum value of any bucket is obviously bounded by numbits-num_bucket_bits
 * 1. I dont need to use double here at all
 * 2. There is still work left to be done
 * 3. Sparse Representation, bias correction ...
 *
 *
 */


template<typename T, size_t num_bucket_bits = 10, typename Hasher = std::hash<T>>
class HyperLog {

private:
    static constexpr size_t num_buckets = (1u << num_bucket_bits) - 1u;
    static constexpr size_t m = 1u << num_bucket_bits;

    std::array<uint8_t, num_buckets> buckets{0};
    size_t count{0};
    Hasher hash;
    double alpha;
public:

    HyperLog() {
        if (m >= 128) {
            alpha = 0.7213 / (1 + 1.079 / ((double) m));
        } else if (m == 16) {
            alpha = 0.673;
        } else if (m == 32) {
            alpha = 0.697;
        } else if (m == 64) {
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

        for (uint64_t i = 0; i < num_bucket_bits; ++i) {
            maske |= HashType{1u} << (num_bits - i - 1);
        }


        //getting the index for the bucket
        HashType bucket_index = maske & hash_value;
        bucket_index = bucket_index >> (num_bits - num_bucket_bits);

        //computing trailing zeros the old fashioned way
        size_t trailing_zeros = 1u;

        for (int i = num_bits - num_bucket_bits - 1; i >= 0; --i) {
            const HashType ml = HashType{1u} << i;
            if ((hash_value & ml) == 0) {
                trailing_zeros++;
            } else {
                break;
            }
        }
        if (buckets[bucket_index] < trailing_zeros)
            buckets[bucket_index] = trailing_zeros;
    }

    size_t get_count() {
        double result;
        double temp = 0;
        for (auto i = 0; i < buckets.size(); ++i) {
            temp += std::pow(2.0, -((double) buckets[i]));
        }
        double eta = (alpha * ((double)m)*((double)m)) / temp;

        int num_zero_buckets = 0;
        for (auto i = 0; i < buckets.size(); ++i) {
            if (buckets[i] == 0)
                num_zero_buckets++;
        }
        if (eta <= (5.0 / 2.0) * ((double) m)) {
            if (num_zero_buckets != 0) {
                result = ((double) m) *
                         std::log(((double) m) / ((double) num_zero_buckets));
            } else {
                result = eta;
            }
        } else if (eta <= (1.0 / 30.0) * std::pow(2.0, 32.0)) {
            result = eta;
        } else {
            result = -std::pow(2.0, 32.0) * std::log(1.0 - eta * std::pow(2.0, -32.0));
        }

        return std::round(result);
    }

    HyperLog &operator+=(const HyperLog &other) {
        //merging the buckets
        for (auto i = 0; i < buckets.size(); ++i) {
            buckets[i] = std::max(buckets[i], other.buckets[i]);
        }
        return *this;
    }


};


#endif //READING_HYPERLOG_H
