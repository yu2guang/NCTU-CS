import sys
import socket

from server import Server
from database import bbsDatabase
import globalVars as GV


def main():
	# initial global variables
	GV.initialize()
	
	# create database
	bbsDB = bbsDatabase()
	GV.DBlock.acquire()
	bbsDB.createTable()	
	GV.DBlock.release()

	# create socket
	PORT = int(sys.argv[1])
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# socket.AF_INET: among servers
	# socket.SOCK_STREAM: seq bitstream, TCP(connection oriented, bi)
	s.bind(("localhost", PORT))
	s.listen(100)

	# receive connect
	while True: 
		conn, addr = s.accept()    
		Server(conn, addr).start()

if __name__ == '__main__':
	main()