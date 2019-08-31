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

void Match::setMaxGames(int games) {
    this->maxGames=games;
}


int Match::getDraws() {
    return draws;
}

int Match::getLosses() {
    return losses;
}

int Match::getWins() {
    return wins;
}


int Match::getElo() {
    //Calculating the elo based on wins,losses,draws
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
            Zobrist::initializeZobrisKeys();
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
                Training::TrainingGame dummy;
                Score result=Utilities::playGame(dummy,first,second,current,print);
                if(result==BLACK_WIN){
                    id=1;
                }else if(result==WHITE_WIN){
                    id=2;
                }
                write(inStream[i][1],(char*)(&id),sizeof(int));

                int id2=3;
                Training::TrainingGame dummy2;
                Score result2=Utilities::playGame(dummy2,second,first,current,false);
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
    std::cout<<std::endl;
    system("echo ' \e[1;31m Engine Match \e[0m' ");
    std::cout<<std::endl;
    printf("%-5s %-5s %-5s","Win", "Loss","Draw");
    printf("\n");
    std::vector<Position>positions;

    Utilities::loadPositions(positions,openingBook);

    int totalGame=0;

    int busy[128]={0};
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
                        wins++;
                    }else if(id==2){
                        losses++;
                    }else{
                        draws++;
                    }
                    busy[p]--;

                    printf("\r");
                    printf("%-5d %-5d %-5d",wins, losses,draws);
                    fflush(stdout);
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
        n--;
    }
    std::cout<<"\n \n Match ended"<<std::endl;
}
