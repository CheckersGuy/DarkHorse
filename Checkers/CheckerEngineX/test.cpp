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

  memstream test(gmlh_netData, gmlh_netSize);

  return 0;
}
