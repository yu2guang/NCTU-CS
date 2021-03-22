import sys
import socket

from client import Client
import globalVars as GV


def main():
	# initial global variables
	GV.initialize()

	# create socket
	HOST = str(sys.argv[1])
	PORT = int(sys.argv[2])
	c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# socket.AF_INET: among servers
	# socket.SOCK_STREAM: seq bitstream, TCP(connection oriented, bi)
	c.connect((HOST, PORT))

	c_thread = Client(c)
	c_thread.start()


if __name__ == '__main__':
	main()