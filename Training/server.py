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
        self.last_update = None
        with open(config_file,"r") as f:
            self.config = yaml.safe_load(f)


    def __del__(self):
        self.stop = True
        if self.thread is not None:
            self.thread.join()

    def get_window_count(self):
        directory = os.fsencode("TrainData")
        max_index = -1000
        for file in os.listdir(directory):
            file_name = os.fsdecode(file)
            if not file_name.startswith(file_name,self.config["name"]+".window"):
                continue
            if not file_name.endswith(".train"):
                continue
            file_name = remove_prefix(file_name,self.config["name"]+".window")
            file_name = remove_postfix(file_name,".train")
            index = int(file_name)
            if index> max_index:
                max_index = index
        return max_index

    def get_latest_network(self):
        directory = os.fsencode("Networks")
        net_file = None
        max_index = -1000
        for file in os.listdir(directory):
            file_name = os.fsdecode(file)
            if not file_name.startswith(self.config["name"]):
                continue
            if not file_name.endswith(".quant"):
                continue
            file_name = remove_postfix(file_name,".quant")
            file_name =remove_prefix(file_name,self.config["name"])
            index = int(file_name)
            if index>max_index:
                max_index =index
                net_file = os.fsdecode(file)

        return net_file


        
    def upload_batch(self, request, context):
        #print("Got a batch of data")
        #print(request.games[0])
        self.counter+=len(request.games)
        print(self.counter)
        return generator_pb2.Response(message="Got the batch")

    def get_last_update(self, request, context):
        return generator_pb2.LastUpdate(timestamp= self.last_update if self.last_update is not None else 0)

    def get_new_network(self, request, context):
        #here needs to be code to get the latest network
        #file = open("Networks/bigagain10.quant","rb")
        #data = file.read()
        #file.close()
        file_name =self.get_latest_network()
        if file_name is None:
            #do something here
            return


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

text="test123.quant"
text = remove_postfix(text,".quant")
text = remove_prefix(text,"test")
print(text)
