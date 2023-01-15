//
// Created by robin on 30.01.22.
//

#include "Thread.h"

void Lockable::aquire() {
    mutex.lock();
}

void Lockable::release() {
    mutex.unlock();
}