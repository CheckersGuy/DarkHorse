//
// Created by robin on 7/26/18.
//

#include "Match.h"

void Match::initializeEngines() {
    first.initialize();
    second.initialize();
}

int Match::getMaxGames() {
    return maxGames;
}

int Match::getTime() {
    return time;
}

void Match::setMaxGames(int games) {
    this->maxGames=games;
}

void Match::setTime(int time) {
    this->time=time;
}

int Match::getDraw() {
    return draw;
}

int Match::getLoss() {
    return loss;
}

int Match::getWin() {
    return win;
}

void Match::start() {
    int outStream[threads][2];
    int inStream[threads][2];

    for(int i=0;i<threads;++i){
        pipe(outStream[i]);
        pipe(inStream[i]);
        pid_t pid =fork();
        if(pid<0){
            exit(EXIT_FAILURE);
        }else if(pid==0){
            //childProcess
            initializeEngines();
            first.setHashSize(21);
            second.setHashSize(21);
            close(outStream[i][1]);
            close(inStream[i][0]);

            while(true){
                Position current;
                read(outStream[i][0],(char*)(&current),sizeof(Position));
                if(current.isEmpty()){
                    break;
                }
                int id=3;
                bool print= false;
                Score result=Utilities::playGame(first,second,current,time,print);
                if(result==BLACK_WIN){
                    id=1;
                }else if(result==WHITE_WIN){
                    id=2;
                }
                write(inStream[i][1],(char*)(&id),sizeof(int));

                int id2=3;
                Score result2=Utilities::playGame(second,first,current,time,false);
                if(result2==BLACK_WIN){
                    id2=2;
                }else if(result2==WHITE_WIN){
                    id2=1;
                }
                write(inStream[i][1],(char*)(&id2),sizeof(int));
            }
            exit(0);
        }
    }

    for(int i=0;i<threads;++i){
        close(outStream[i][0]);
        close(inStream[i][1]);
        fcntl(inStream[i][0],F_SETFL,O_NONBLOCK|O_RDONLY);
    }

    std::vector<Position>positions;

    Utilities::loadPositions(positions,openingBook);

    int totalGame=0;

    int busy[threads]={0};
    int idx=0;

    while(totalGame<maxGames){
        for(int p=0;p<threads;++p){
            if(busy[p]==0){
                write(outStream[p][1],(char*)(&positions[idx]),sizeof(Position));
                busy[p]=2;
                idx++;
                idx=idx%positions.size();
            }else{
                int id;
                int buf =read(inStream[p][0],(char*)(&id),sizeof(int));
                if(buf!=-1){
                    totalGame++;
                    if(id==1){
                        win++;
                    }else if(id==2){
                        loss++;
                    }else{
                        draw++;
                    }
                    busy[p]--;
                    printf("%d-%d=%d \n",win,loss,draw);
                }
            }
        }
    }
    //Once we have done all the work, stop the children
    for(int i=0;i<threads;++i){
        Position empty;
        write(outStream[i][1],(char*)(&empty),sizeof(Position));
    }


    int n=threads;
    while(n>0){
        int status;
        wait(&status);
        std::cout<<"Child stopped with status: "<<status<<std::endl;
        n--;
    }
}

int Match::getElo() {
    //Calculating the elo based on win,loss,draw
    return 0;
}

void Match::setOpeningBook(std::string book) {
    this->openingBook=book;
}

std::string& Match::getOpeningBook() {
    return openingBook;
}

void Match::setNumThreads(int threads) {
    this->threads=threads;
}

int Match::getNumThreads() {
    return threads;
}