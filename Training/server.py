import grpc
import generator_pb2
import generator_pb2_grpc
import time
from concurrent import futures
from rich.live import Live
from rich.table import Table
import threading

#to be added
class Generator(generator_pb2_grpc.GeneratorServicer):

    def __init__(self):
        self.counter =0
        self.thread = None
        self.stop = False

    def __del__(self):
        self.stop = True
        if self.thread is not None:
            self.thread.join()
        
    def upload_batch(self, request, context):
        #print("Got a batch of data")
        #print(request.games[0])
        self.counter+=len(request.games)
        print(self.counter)
        return generator_pb2.Response(message="Got the batch")

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
    server.add_insecure_port("[::]:50051")
    print("Server starting")
    server.start()
    server.wait_for_termination()

server()
