import subprocess
import multiprocessing
import threading
from enum import Enum,unique
from random import randrange
from rich.live import Live
from rich.table import Table
from rich.align import Align
import time
import os
import random
import generator_pb2
#Needs to be reworked using a multiprocessing pool


#grpc service probably goes here


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
print_lock = multiprocessing.Lock()
write_lock = multiprocessing.Lock()
games = multiprocessing.Manager().list()
start_event = multiprocessing.Event()

class Interface:
    
    def __init__(self):
        #grpc-stuff
        self.current_game = []
        self.process = None
        self.openings=[]
        self.engine =EngineState()
        self.hash_size =21
        self.max_games =100000
        self.start_time =None
        self.time_per_move =100
        self.adjudication=0
        self._parallelism = 1
        self.network_file = ""


       
    def time_convert(self,time_seconds):
         mins =time_seconds//60
         sec =time_seconds%60
         hours = mins//60
         mins  = mins%60
         return hours,mins,sec


    def get_table(self):
        table =Table(title="Generating Games")
        time_lapsed = time.time()-self.start_time
        table.add_column("NumGames")
 
        table.add_column("Parallelism")
        table.add_column("MaxGames")
        table.add_column("HashSize")
        table.add_column("TimeLapsed")
        table.add_column("GamesPerSec")
        hours,mins,sec = self.time_convert(time_lapsed)
        num_games_sec ="{:.2f}".format(counter.value/time_lapsed)
        table.add_row(str(counter.value),str(self.parallelism),str(self.max_games),str(self.hash_size),"{0}:{1}:{2}".format(int(hours),int(mins),int(sec)),num_games_sec)
        table = Align.center(table,vertical = "middle")
        return table

    def print_table(self):
        with Live(self.get_table(),refresh_per_second=4) as live:
            while counter.value<self.max_games:
                time.sleep(0.1)
                live.update(self.get_table())

    
    

 
    def process_stream(self,index):
        print("Waiting for the event")
        start_event.wait()
        print("Finished waiting for the event")
        process = subprocess.Popen(["../cmake-build-debug/MainEngine","--selfplay","--network Networks/{}".format(self.network_file)],stdout = subprocess.PIPE,stdin=subprocess.PIPE,stderr = subprocess.PIPE)
        random.seed((os.getpid()*int(time.time()))%123456789)
        #channel = grpc.insecure_channel("localhost:50051")
        global counter
        while True:
            if counter.value>=self.max_games:
                self.send_termination_command(process)
                break

            if self.engine.state == States.DEFAULT:
                self.load_network_command("Networks/{}".format(self.network_file),process)
                self.send_settings_command(process)
                self.engine.state = States.INIT
            
            if self.engine.state == States.INIT:
                #print("Refreshed network")
                self.send_play_command(self.pick_opening(),process)
                self.engine.state = States.PLAYING_GAME

            line  = process.stdout.readline()
            if not line:
                break
            line =line.decode().rstrip()
            if line =="game_start":
                self.current_game=[]
                self.engine.state = States.RECV_GAME

            if line =="game_end":
                if len(self.current_game)>1:
                    write_lock.acquire()
                    games.append(self.current_game)
                    write_lock.release()
                self.current_game.clear()
                self.engine.state = States.INIT
                with counter.get_lock():
                    counter.value+=1
            #print(line)
            #getting the games
            if self.engine.state  == States.RECV_GAME:
                if line !="game_start" and line!="game_end":
                    self.current_game.append(line)

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

    def load_network_command(self,file,stream):
        cmd="loadnetwork!"+file+"\n"
        stream.stdin.writelines([cmd.encode()])
        stream.stdin.flush()

    def send_termination_command(self,stream):
        cmd="terminate\n"
        stream.stdin.writelines([cmd.encode()])
        stream.stdin.flush()

    def send_settings_command(self,stream):
        cmd="settings!"+str(self.time_per_move)+"!"+str(self.hash_size)+"!"+str(self.adjudication)+"\n"
        stream.stdin.writelines([cmd.encode()])
        stream.stdin.flush()

    def start(self):
        p = multiprocessing.Pool(self.parallelism)
        results = p.map_async(self.process_stream,range(self.parallelism))
        #Only executing in main process from here on
        self.start_time =time.time()
        thread = threading.Thread(target=self.print_table)
        thread.start()
        self.generate()
        thread.join()
        results.get()
    

    def generate(self):
        start_event.set()
        while True:
            time.sleep(0.1)
            write_lock.acquire()
            if len(games)<self.max_games:
                write_lock.release()
                continue
            batch = generator_pb2.Batch()
            for g in games:
                game = generator_pb2.Game()
                game.start_position = g[0]
                game.move_indices.extend([int(value) for value in g[1:]])
                batch.games.append(game)
            data = batch.SerializeToString()
            with open("window.train","wb") as file:
                file.write(data)
 
            write_lock.release()
            break;


                

    @property
    def parallelism(self):
        return self._parallelism

    #settter
    @parallelism.setter
    def parallelism(self,value):
        self._parallelism = value
        #setting the events

  


interface = Interface()
interface.time_per_move = 50
interface.parallelism = 4
interface.hash_size =22
interface.max_games =10000
interface.network_file="testing6.quant"
interface.read_openings("Positions/train13.book")
interface.start()

