

#include <cmath>
inline double sigmoid(double x) { return (std::exp(x) / (std::exp(x) + 1.0)); }

inline double logit(double x) { return std::log(x / (1.0 - x)); }
