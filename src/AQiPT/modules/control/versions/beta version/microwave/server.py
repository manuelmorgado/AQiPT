# server for the TCP/IP connection between server-client

import socket as skt # importing library for the tcp/ip commands

s = skt.socket()

ip_address = '127.0.0.1' #ip address of server
gate = 9999 #gate address

print('Socket created!')

skt.bind((ip_address, gate)) #creating biding
s.listen(1) #nr of listening clients

print('Waiting for connection!')

while True:

	c, addr  = skt.accept() #accepting clients in qeue
	print("Connection with ", addr, " done!")

	c.send(bytes("Welcome! ", 'utf-8')) #sending admision message
	
	out_data =  #data to be send from server (lab comp) to client (producer)
	"""
	Example for out_data:

		"Gaussian_pulse(std, ampl, phase, start)

		Square_puilse(twidth, tstart, phase, amp, freq)"
	"""

	c.send(byter(out_data, 'utf-8')) #data sending


	c.close()
