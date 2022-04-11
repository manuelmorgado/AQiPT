# client for the TCP/IP connection between server-client

import socket as skt # importing library for the tcp/ip commands
from aqipt.pulses import *

c = skt.socket()

ip_address = '127.0.0.1'
gate = 9999

c.connect ((ip_address, gate))

in_data = c.recv(1024).decode()

for line in in_data:
	if line == '\n':
		pass
	else
		print('Calling the pulse command!')
		exec(str(line))

