//
// Created by robin on 4/2/19.
//

#ifndef CHECKERENGINEX_THREAD_H
#define CHECKERENGINEX_THREAD_H

//Will continue all the Thread-stuff (including threadpool)
//Needed for YBWC

#include <memory>
#include <thread>

struct SplitPoint {

};


struct ThreadBase {

    virtual void search() const = 0;
};


#endif //CHECKERENGINEX_THREAD_H
