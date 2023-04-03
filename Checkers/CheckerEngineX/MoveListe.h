#ifndef CHECKERSTEST_MOVELISTE_H
#define CHECKERSTEST_MOVELISTE_H

#include "Move.h"
#include "types.h"
#include "MovePicker.h"
#include <array>
#include "Line.h"
#include "Position.h"

class MoveListe {

private:
    int moveCounter = 0;
public:
    std::array<Move, 40> liste;

    int length() const;

    void add_move(Move next);

    void sort(Position current, Depth depth, int ply, Move ttMove,Move previous,Move previous_own, int start_index);

    bool is_empty() const;

    const Move &operator[](int index) const;

    Move &operator[](int index);

    bool put_front(Move other);

	bool put_front(int start_index, Move other);

    void remove(Move move);

    void reset();


    auto begin() {
        return liste.begin();
    }

    auto end() {
        auto it = liste.begin();
        std::advance(it, moveCounter);
        return it;
    }

    MoveListe &operator=(const MoveListe &other);

    template<MoveType type>
    inline void visit(uint32_t &maske, uint32_t &next) {
        add_move(Move{maske, next});
    };

    template<MoveType type>
    inline void visit(uint32_t &from, uint32_t &to, uint32_t &captures) {
        add_move(Move{from, to, captures});
    };



};

inline bool MoveListe::is_empty() const {
    return this->moveCounter == 0;
}

inline const Move &MoveListe::operator[](int index) const {
    return liste[index];
}

inline Move &MoveListe::operator[](int index) {
    return liste[index];
}

inline void MoveListe::add_move(Move next) {
    liste[this->moveCounter++] = next;
}

inline int MoveListe::length() const {
    return moveCounter;
}



#endif //CHECKERSTEST_MOVELISTE_H
