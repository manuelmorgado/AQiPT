from tkinter import *

root = Tk() #original widget (e.g., window). In tkinter everything is a widget


def myClick():
	aLabel = Label(root, text="This is A label") #creating a label in the root object with that text
	aLabel.grid(row=0, column=10)
	# aLabel.pack() #showing into the screen

# bLabel = Label(root, text="This is B label") #creating a label in the root object with that text
# bLabel.grid(row=100, column=100)


aButton = Button(root, text="click", command=myClick).grid(row=0, column=300)
# aButton.pack()

root.mainloop()