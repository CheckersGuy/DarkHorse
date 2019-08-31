//
// Created by robin on 9/17/18.
//


#include "Line.h"



Line::Line(const Line &other) {
    this->counter=other.counter;
    for(int i=0;i<other.counter;++i){
        this->myArray[i]=other.myArray[i];
    }
}

int Line::length()const {
    return counter;
}

void Line::addMove(Move move) {
    this->myArray[counter++] = move;
}

void Line::concat(Move best, Line& line) {

    myArray[0]=best;
    for (int i = 0; i < line.length(); ++i) {
        myArray[i+1]=line.myArray[i];
    }
    this->counter =line.counter+1;
}

std::string Line::toString() const {
    std::string current;
    for (int k = 0; k <length(); ++k) {
        current+=" ("+ std::to_string(k) + ") " + "[" + std::to_string(myArray[k].getFrom()) +"|";
        current+= std::to_string(myArray[k].getTo()) + "]";
    }
    return current;
}

std::ostream& operator<<(std::ostream& stream, Line& line){
    stream<<line.toString();
    return stream;
}


Move Line::getFirstMove() const {
    return myArray[0];
}

void Line::clear() {
    this->counter = 0;
}


bool Line::operator==(const Line &other)const {
    if (other.length() != this->length())
        return false;

    for (int i = 0; i < other.length(); ++i) {
        if (myArray[i] != other.myArray[i])
            return false;
    }

    return true;
}


bool Line::operator!=(const Line &other)const  {

    return *this != other;
}

Line& Line::operator=(const Line &other) {
    for(int i=0;i<other.length();++i){
        this->myArray[i]=other.myArray[i];
    }
    this->counter=other.counter;
    return *this;
}

Move Line::operator[](int index)const {
    return this->myArray[index];
}

Move& Line::operator[](int index) {
    return this->myArray[index];
}
Move* Line::begin() {
    return &myArray[0];
}

Move* Line::end() {
    return &myArray[counter];
}