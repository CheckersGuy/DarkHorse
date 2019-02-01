//
// Created by robin on 1/25/19.
//

#ifndef CHECKERENGINEX_FIXPOINT_H
#define CHECKERENGINEX_FIXPOINT_H

#include <iomanip>

template<class T, size_t scal>
class FixPoint {
private:

    T m = 0;

public:

    static constexpr T scalingFactor = 1 << scal;

    FixPoint() = default;


    FixPoint<T, scal>(const FixPoint<T, scal> &other) {
        this->m = other.m;
    }

    FixPoint<T, scal>(FixPoint<T, scal> &&other) {
        this->m = other.m;
    }

    template<class U>
    FixPoint<T, scal>(U other) {
        this->m = other * scalingFactor;
    }

    template<class U>
    FixPoint<T, scal> &operator=(U &&other) {
        if constexpr(std::is_same<U, FixPoint<T, scal>>::value) {
            this->m = other.m;
        } else {
            this->m = other * scalingFactor;
        }
        return *this;
    };


    template<class U>
    FixPoint<T, scal> &operator+=(const U other) {
        if constexpr(std::is_same<U, FixPoint<T, scal>>::value) {
            this->m += other.m;
        } else {
            this->m += scalingFactor * other;
        }

        return *this;
    };

    template<class U>
    FixPoint<T, scal> operator+(const U other) {
        FixPoint<T, scal> newValue;
        if constexpr(std::is_same<U, FixPoint<T, scal>>::value) {
            newValue.m = this->m;
            newValue.m += other.m;
        } else {
            newValue.m = this->m;
            newValue.m += other * scalingFactor;
        }
        return newValue;
    };


    template<class U>
    FixPoint<T, scal> &operator-=(U other) {
        if constexpr(std::is_same<U, FixPoint<T, scal>>::value) {
            this->m -= other.m;
        } else {
            this->m -= other * scalingFactor;
        }

        return *this;
    };

    template<class U>
    FixPoint<T, scal> operator-(const U other) {
        FixPoint<T, scal> newValue;
        if constexpr(std::is_same<U, FixPoint<T, scal>>::value) {
            newValue.m = this->m;
            newValue.m -= other.m;
        } else {
            newValue.m = this->m;
            newValue.m -= other * scalingFactor;
        }

        return newValue;
    };

    template<class U>
    FixPoint<T, scal> &operator*=(const U other) {
        if constexpr(std::is_same<U, FixPoint<T, scal>>::value) {
            this->m = (this->m * other.m) / scalingFactor;
        } else {
            this->m = this->m * other;
        }

        return *this;
    };

    template<class U>
    FixPoint<T, scal> operator*(const U other) {
        FixPoint<T, scal> newValue;
        if constexpr(std::is_same<U, FixPoint<T, scal>>::value) {
            newValue.m = (this->m * other.m) / scalingFactor;
        } else {
            newValue.m = this->m * other;
        }
        return newValue;
    };

    template<class U>
    FixPoint<T, scal> &operator/=(const U other) {
        if constexpr(std::is_same<U, FixPoint<T, scal>>::value) {
            this->m = (this->m * scalingFactor) / (other.m);
        } else {
            this->m = this->m / (other);
            return *this;
        }

        return *this;
    };

    template<class U>
    FixPoint<T, scal>operator/(const U other) {
        FixPoint<T, scal> newValue;
        if constexpr(std::is_same<U, FixPoint<T, scal>>::value) {
            newValue.m = (this->m * scalingFactor) / (other.m);
        } else {
            newValue.m = this->m / (other);
        }
        return newValue;
    };


    template<class U>
    U as() const {
        return static_cast<U>(m) / static_cast<U>(scalingFactor);
    }

    T round() {
        return (m + scalingFactor / 2) / scalingFactor;
    }

    template<class U>
    bool operator==(const U &&other) {
        if constexpr(std::is_same<U, FixPoint<T, scal>>::value) {
            return other.m == this->m;
        } else {
            T temp = scalingFactor * other;
            return temp == this->m;
        }
    }

    template<class U>
    bool operator<(const U &&other) {
        if constexpr(std::is_same<U, FixPoint<T, scal>>::value) {
            return this->m < other.m;
        } else {
            T temp = scalingFactor * other;
            return this->m < temp;
        }
    }

    template<class U>
    bool operator<=(const U &&other) {
        if constexpr(std::is_same<U, FixPoint<T, scal>>::value) {
            return this->m <= other.m;
        } else {
            T temp = scalingFactor * other;
            return this->m <= temp;
        }
    }

    template<class U>
    bool operator>(const U &&other) {
        if constexpr(std::is_same<U, FixPoint<T, scal>>::value) {
            return this->m > other.m;
        } else {
            T temp = scalingFactor * other;
            return this->m > temp;
        }
    }

    template<class U>
    bool operator>=(const U &&other) {
        if constexpr(std::is_same<U, FixPoint<T, scal>>::value) {
            return this->m >= other.m;
        } else {
            T temp = scalingFactor * other;
            return this->m >= temp;
        }
    }


    friend std::ostream &operator<<(std::ostream &stream, const FixPoint &other) {
        stream << other.as<double>();
        return stream;
    }
};

#endif //CHECKERENGINEX_FIXPOINT_H
