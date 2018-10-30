//
// Created by robin on 7/26/18.
//

#include "Generator.h"
#include "Engine.h"
#include  <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include  <sys/types.h>
#include "fcntl.h"
#include "Utilities.h"


void Generator::setMaxGames(int games) {
    this->maxGames = games;
}

void Generator::setThreads(int threads) {
    this->threads = threads;
}

void Generator::clearBuffer() {
    std::ofstream stream(output, std::ios::binary | std::ios::app);
    if (!stream.good()) {
        std::cerr << "Error file error" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (TrainingGame &game : buffer) {
        stream.write((char *) (&game.result), sizeof(Score));
        int length = game.positions.size();
        stream.write((char *) (&length), sizeof(int));
        stream.write((char *) &game.positions[0], sizeof(Position) * length);
    }
    stream.close();
    buffer.clear();
}

int Generator::getMaxGames() {
    return maxGames;
}

int Generator::getTime() {
    return time;
}

int Generator::getThreads() {
    return threads;
}

void Generator::setTime(int time) {
    this->time = time;
}


void Generator::start() {
    int outStream[threads][2];
    int inStream[threads][2];

    for (int i = 0; i < threads; ++i) {
        pipe(outStream[i]);
        pipe(inStream[i]);
        pid_t pid = fork();
        if (pid < 0) {
            exit(EXIT_FAILURE);
        } else if (pid == 0) {
            engine.initialize();
            engine.setHashSize(18);
            close(outStream[i][1]);
            close(inStream[i][0]);

            while (true) {
                Position current;
                read(outStream[i][0], (char *) (&current), sizeof(Position));
                if (current.isEmpty()) {
                    break;
                }
                TrainingGame game;
                Score result = Utilities::playGame(game, engine, engine, current, time, false);
                game.result = result;
                int length = game.positions.size();
                if(result==INVALID)
                    length=0;
                write(inStream[i][1], (char *) (&game.result), sizeof(Score));
                write(inStream[i][1], (char *) (&length), sizeof(int));
                write(inStream[i][1], (char *) (&game.positions[0]), sizeof(Position) * length);

            }
            exit(0);
        }
    }

    for (int i = 0; i < threads; ++i) {
        close(outStream[i][0]);
        close(inStream[i][1]);
        fcntl(inStream[i][0], F_SETFL, O_NONBLOCK | O_RDONLY);
    }

    std::vector<Position> positions;

    Utilities::loadPositions(positions, book);

    int totalGame = 0;

    bool busy[threads] = {false};
    int idx = 0;

    while (totalGame < maxGames) {
        for (int p = 0; p < threads; ++p) {
            if (!busy[p]) {
                write(outStream[p][1], (char *) (&positions[idx]), sizeof(Position));
                busy[p] = true;
                idx++;
                idx = idx % positions.size();
            } else {
                Score result;
                int buf = read(inStream[p][0], (char *) (&result), sizeof(Score));
                if (buf != -1) {
                    std::cout << "Counter: " << totalGame << std::endl;
                    TrainingGame game;
                    int length;
                    while (read(inStream[p][0], (char *) (&length), sizeof(int)) == -1);
                    game.positions.reserve(length + 1);
                    for (int k = 0; k < length; ++k) {
                        Position current;
                        while (read(inStream[p][0], (char *) (&current), sizeof(Position)) == -1);
                        game.add(current);
                    }
                    game.result = result;
                    busy[p] = false;
                    totalGame++;
                    buffer.emplace_back(game);
                    if ((buffer.size() + 1) % 1000 == 0) {
                        clearBuffer();
                        std::cout << "Buffer cleared" << std::endl;
                    }
                }
            }
        }
    }
    //Once we have done all the work, stop the children
    for (int i = 0; i < threads; ++i) {
        Position empty;
        write(outStream[i][1], (char *) (&empty), sizeof(Position));
    }


    int n = threads;
    while (n > 0) {
        int status;
        wait(&status);
        std::cout << "Child stopped with status: " << status << std::endl;
        n--;
    }
    clearBuffer();
}