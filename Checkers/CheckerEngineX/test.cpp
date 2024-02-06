#include "Bits.h"
#include "incbin.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <emmintrin.h>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <random>

INCBIN(mlh_net, "mlh2.quant");

int main() {

  std::cout << (int)gmlh_netData[303] << std::endl;

  return 0;
}
