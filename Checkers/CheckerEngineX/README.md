# Checkers
CheckersEngine

Did finally merge with searchExp which is definently a good thing. I am kinda
procrastinating because I have no clue how to start with the moveGenerator stuff. There are still so many things to be done until I can start with
something more intresting.

Things still to be done

* change the moveGenerator for jumps and possbile for silent moves too
* change how I order moves. Change which algorithm I use (Create a class called MPicker which does the sorting)
* 'Empower' my understanding of search (alpha-beta and so on) and i want to change everything to failSoft instead of continue using failHard
* failSoft is widely regarded as being superior to failHard and with failSoft I (personally) have no problems implementing aspiration windows and some of the other stuff
the only downside with failSoft is that it requires somewhat more code which means that it is slightly more complex but the tradeoff between better search and complexity
is ok
* After doing all this (probably including the implementation of aspiration search) I will return to the whole weights issue which is an entirely seperate issue
There were a couple of bugs. For example collecting training data needs to be reimplemented because I was stupid and forgot to store the player to move inside the Position struct.
However, I was quite unsure about the whole weights stuff but finding this bug gave me hope that it will finally work