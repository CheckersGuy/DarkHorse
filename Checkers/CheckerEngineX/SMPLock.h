//
// Created by robin on 11/20/18.
//

#ifndef CHECKERENGINEX_SMPLOCK_H
#define CHECKERENGINEX_SMPLOCK_H

#include <atomic>

class SMPLock {
private:
    std::atomic_flag flag{ATOMIC_FLAG_INIT};
public:

    void lock();

    void unlock();
};

inline void SMPLock::lock() {
    while (flag.test_and_set(std::memory_order_acquire));
}

inline void SMPLock::unlock() {
    flag.clear(std::memory_order_release);
}


#endif //CHECKERENGINEX_SMPLOCK_H
