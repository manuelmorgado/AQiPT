#Atomic Quantum information Processing Tool (AQIPT) - Datalogger module

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Created: 2021-04-08
# Last update: 2023-01-15

import logging
import tkinter as tk

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler
handler = logging.FileHandler('app.log')
handler.setLevel(logging.DEBUG)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(handler)

# Use the logger in your code
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')





class LoggerWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Real-time Logger")
        self.geometry("400x300")

        # Create a Text widget to display the log messages
        self.log_text = tk.Text(self)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create a handler that will write log messages to the Text widget
        self.log_handler = TextHandler(self.log_text)
        self.log_handler.setLevel(logging.DEBUG)

        # Add the handler to the logger
        self.logger.addHandler(self.log_handler)

    def run(self):
        self.mainloop()

class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, f'{msg}\n')
        self.text_widget.configure(state='disabled')

# Create an instance of the LoggerWindow and run it
win = LoggerWindow()
win.run()

# Use the logger in your code
win.logger.debug('This is a debug message')
win.logger.info('This is an info message')
win.logger.warning('This is a warning message')
win.logger.error('This is an error message')
win.logger.critical('This is a critical message')
