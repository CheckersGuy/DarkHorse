import grpc
import generator_pb2
import generator_pb2_grpc
import time
from concurrent import futures
from rich.live import Live
from rich.table import Table
import threading
import os
import yaml
import trainer
from datetime import datetime
def remove_prefix(text,prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def remove_postfix(text,postfix):
    if text.endswith(postfix):
        return text[:len(text)-len(postfix)]
    return text


#to be added
class Generator(generator_pb2_grpc.GeneratorServicer):

    def __init__(self,config_file="test.yaml"):
        self.counter =0
        self.thread = None
        self.stop = False
        self.last_update = 1000
        with open(config_file,"r") as f:
            self.config = yaml.safe_load(f)
        self.window_games =generator_pb2.Batch()



    def __del__(self):
        self.stop = True
        if self.thread is not None:
            self.thread.join()

    def create_random_network(self):
        print("Creating random network")
        network = trainer.LitMLP.Network(self.config["network"])
        counter = self.get_window_count()+1
        network.save_quantized("Networks/{}{}.quant".format(self.config["name"],counter))
        

    def generate_next(self,window_count):
        if window_count<0:
            #do not have any data
            return
        trainer.train_network(self.config["name"],self.get_window_count()+1,train_file="{}.window{}.train".format(self.config["name"],window_count))


    #below method not working properly
    def get_window_count(self):
        directory = os.fsencode("TrainData")
        max_index = -1
        for file in os.listdir(directory):
            file_name = os.fsdecode(file)
            if not file_name.endswith(".train"):
                continue
            if not file_name.startswith(self.config["name"]+".window"):
                continue
            file_name = remove_prefix(file_name,self.config["name"]+".window")
            file_name = remove_postfix(file_name,".train")
            index = int(file_name)
            if index > max_index:
                max_index = index
        return max_index

    def get_latest_network(self):
        directory = os.fsencode("Networks")
        net_file = None
        max_index = -1
        for file in os.listdir(directory):
            file_name = os.fsdecode(file)
            orig_name = file_name
            if not file_name.startswith(self.config["name"]):
                continue
            if not file_name.endswith(".quant"):
                continue
            file_name = remove_postfix(file_name,".quant")
            file_name =remove_prefix(file_name,self.config["name"])
            index = int(file_name)
            if index>max_index:
                max_index =index
                net_file = orig_name

        return net_file


        
    def upload_batch(self, request, context):
        self.counter+=len(request.games)
        self.window_games.games.extend(request.games)
        length = len(self.window_games.games)
        if length >= self.config["window_size"]:
            data = self.window_games.SerializeToString()
            window_counter = self.get_window_count()+1
            file = open("TrainData/"+self.config["name"]+".window"+str(window_counter)+".train","wb")
            file.write(data)
            file.close()
            print("Filled up the window")
            self.window_games.ClearField("games")
            #starting training of a new network
            print("Start Training")
            #with trainer.stdout_redirected():
            self.generate_next(window_counter)
            #print("Ended training")
            dt_tim = datetime.now()
            self.last_update=int(round(dt_tim.timestamp()))
        return generator_pb2.Response(message="Got the batch")

    def get_last_update(self, request, context):
        return generator_pb2.LastUpdate(timestamp= self.last_update)

    def get_new_network(self, request, context):
        file_name =self.get_latest_network()
        print("Found network: ", file_name)
        if file_name is None:
            #creating a random network
            file_name =self.config["name"]+str(0)+".quant"
            self.create_random_network()


        print("Name:",file_name)
        file = open("Networks/"+file_name,"rb")
        data = file.read()
        file.close()
        net_msg = generator_pb2.Network()
        net_msg.data = data
        return net_msg

    def get_table(self):
        table = Table(title="Getting games")
        table.add_column("NumGames")
        table.add_row(str(self.counter))
        return table

    def print_table(self):
        with Live(self.get_table(),refresh_per_second=4) as live:
            while not self.stop:
                time.sleep(0.4)
                live.update(self.get_table())

    def start_ui_thread(self):
        self.thread = threading.Thread(target=self.print_table)
        self.thread.start();




def server():
    generator =Generator()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    generator.start_ui_thread()
    print("Test")
    generator_pb2_grpc.add_GeneratorServicer_to_server(generator,server)
    server.add_insecure_port("[::]:50052")
    print("Server starting")
    server.start()
    server.wait_for_termination()

if __name__=="__main__":
    server()


