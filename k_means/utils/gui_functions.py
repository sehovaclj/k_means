# functions to import into class
import tkinter as tk


def create_greeting(root_window, frame_counter):
    greeting_frame = tk.Frame(master=root_window)
    greeting_label = tk.Label(master=greeting_frame,
                              text="""Hello! 
                Please input desired parameters for the K-means Clustering simulation.
                """)
    greeting_label.grid(row=frame_counter)
    greeting_frame.grid(row=frame_counter, column=1)
    frame_counter += 1
    return frame_counter
