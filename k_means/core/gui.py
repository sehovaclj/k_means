import tkinter as tk
from k_means.utils.gui_functions import create_greeting


class InteractiveParameters:
    """A class which is created on main run. Intended to interact with user.
        Main goal is to get desired parameters from user.

    Args:
        something_here: explain something here too

    """

    def __init__(self):
        self.root = tk.Tk()
        self.labels = ["Num. Clusters",
                       "Num. Distributions",
                       "Num. Samples",
                       "Epsilon Convergence",
                       "Max. Iterations",
                       "Add Noise",
                       "Pause Length",
                       "Seed"]
        self.frame_counter = 0
        create_greeting(self.root, self.frame_counter)
        # add a greeting
        # self.greeting_frame = tk.Frame(master=self.root)
        # self.greeting_label = tk.Label(master=self.greeting_frame,
        #                                text="""Hello!
        #     Please input desired parameters for the K-means Clustering simulation.
        #     """)
        # self.greeting_label.grid(row=0)
        # self.greeting_frame.grid(row=0, column=1)
        # get parameters
        self.parameters_frame = tk.Frame(master=self.root)
        self.counter = 0
        for parameter in self.labels:
            parameter_label = tk.Label(master=self.parameters_frame,
                                       text=parameter)
            parameter_label.grid(row=self.counter)
            parameter_renamed = parameter.lower().replace('.', '').replace(' ', '_')
            setattr(self, 'entry_' + parameter_renamed, tk.Entry(master=self.parameters_frame))
            getattr(self, 'entry_' + parameter_renamed).grid(row=self.counter, column=1)
            self.counter += 1
        self.parameters_frame.grid(row=1, column=1)
        # assign button commands to run or quit
        self.buttons_frame = tk.Frame(master=self.root)
        self.run_button = tk.Button(master=self.buttons_frame,
                                    text='Run',
                                    command=self.root.destroy)
        self.run_button.grid(row=0,
                             column=1,
                             padx=10,
                             pady=10)
        self.quit_button = tk.Button(master=self.buttons_frame,
                                     text='Quit',
                                     command=self.root.destroy)
        self.quit_button.grid(row=0,
                              column=2,
                              padx=10,
                              pady=10)
        self.buttons_frame.grid(row=2, column=1)
        self.root.title('GUI for K-means')
        self.root.attributes("-topmost", True)
        self.root.mainloop()
