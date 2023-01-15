import subprocess
import multiprocessing
from enum import Enum,unique
from random import randrange
import time


#Needs to be reworked using a multiprocessing pool
@unique
class States(Enum):
    DEFAULT = 0 # nothing happening
    RECV_GAME= 1 #game was generated and client send game_start
    PLAYING_GAME=2 #client is playing a game
    INIT = 3 #Engine has been initialized


class EngineState:

    def __init__(self):
        self._state =States.DEFAULT
        self.opening_index =0 

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self,value):
        self._state = value


counter = multiprocessing.Value("i",0)
class Interface:
    
    def __init__(self):
        self.process = None
        self.openings=[]
        self.engines =[]
        self.hash_size =21
        self.time_per_move =100
        self.adjudication=10
        self._parallelism = 1
        self.network = None

    @property
    def parallelism(self):
        return self._parallelism


    @parallelism.setter
    def parallelism(self,value):
        self._parallelism =value
        self.engines =[EngineState() for i in range(self._parallelism)]

 
    def process_stream(self,index):
        process = subprocess.Popen(["./MainEngine","--selfplay","--network ../cmake-build-debug/bigagain10.quant"],stdout = subprocess.PIPE,stdin=subprocess.PIPE,stderr = subprocess.PIPE)
        global counter
        while True:
            if self.engines[index].state == States.INIT:
                self.send_play_command(self.pick_opening(),process)
                self.engines[index].state = States.PLAYING_GAME

            if self.engines[index].state == States.DEFAULT:
                self.send_settings_command(process)
                self.engines[index].state = States.INIT

            line  = process.stdout.readline()
            if not line:
                break
            line =line.decode().rstrip()
            if line =="game_start":
                self.engines[index].state = States.RECV_GAME

            if line =="game_end":
                self.engines[index].state = States.INIT
                with counter.get_lock():
                    counter.value+=1
                print(counter.value)
            #print(line)
        process.wait()



    def read_openings(self,file_name):
        with open(file_name) as file:
            self.openings=[line.rstrip() for line in file]

    def pick_opening(self):
        index = randrange(len(self.openings))
        return self.openings[index]

    def send_play_command(self,fen_string,stream):
        cmd = "playgame!"+fen_string+"\n"
        stream.stdin.writelines([cmd.encode()])
        stream.stdin.flush()

    def send_settings_command(self,stream):
        cmd="settings!"+str(self.time_per_move)+"!"+str(self.hash_size)+"!"+str(self.adjudication)+"\n"
        stream.stdin.writelines([cmd.encode()])
        stream.stdin.flush()

    def start(self):
        p = multiprocessing.Pool(self.parallelism)
        p.map(self.process_stream,range(self.parallelism))

  


interface = Interface()
interface.time_per_move = 10
interface.parallelism = 14
interface.hash_size =21
interface.read_openings("Positions/train9.book")
interface.start()



