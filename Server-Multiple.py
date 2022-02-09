import threading
import socket
import os
 
from datetime import datetime

from threading import Thread
clients = set()
clients_lock = threading.Lock()

def listener(client, address):
    global a
    print ("Accepted connection from: ", address)
    with clients_lock:
       clients.add(client)
    try:
        while True:
            data = client.recv(1024)
            if not data:
                break
            else :
                b = datetime.now()
                t=str(int((b-a).total_seconds() * 1000))
                # print (repr(data))
                # file1 = open("monitor.txt", "a+")  # append mode
                # file1.write(t+","+repr(data)+" "+str(address[0])+"\n")
                # file1.close()
                with clients_lock:
                    for c in clients:
                        c.sendall(data)
    finally:
        with clients_lock:
            clients.remove(client)
            client.close()


# host = socket.gethostname()
host = 'localhost'
port = 1233       
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((host,port))
s.listen(3)
th = []

while True:
    print ("Server is listening for connections...")
    client, address = s.accept()
    a = datetime.now()
    th.append(Thread(target=listener, args = (client,address)).start())
    

s.close()