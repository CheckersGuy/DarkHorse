//
// Created by robin on 2/17/19.
//

#include "Index.h"

namespace Index {


    Index pascalD[33][33];

    void initialize() {
        for (int i = 0; i <= 32; ++i) {
            pascalD[i][0] = 1;
            pascalD[i][i] = 1;
        }

        for (int i = 1; i <= 32; ++i) {
            for (int j = 1; j < i; ++j) {
                pascalD[i][j] = pascalD[i - 1][j - 1] + pascalD[i - 1][j];
            }
        }
    }


}