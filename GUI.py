# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox
from Analyzer import Analyzer
from observer import Publisher, Subscriber
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import ImageTk, Image
import asyncio
import threading
import time


class Model(Publisher):
    def __init__(self, events):
        super().__init__(events)
        self.analyzer = Analyzer()
        self.input_data = None
        self.datadict = None
        self.dataarray = None
        self.pca = None
        self.fig = None
        self.ax = None

        self.input_path = None
        self.output_path = None

    def clearData(self):
        self.input_data = None
        self.datadict = None
        self.dataarray = None

    def dataLoadedAndAvailable(self):
        if self.input_data == None:
            return False
        else:
            return True

    #import the input_data into the input_data framework
    async def prepare_data(self):
        self.datadict, self.dataarray = await preprare_data(self.input_data)


    #reads the files
    async def read_data(self):
        if self.input_path == None:
            messagebox.showerror( 'Error', 'no path!')
            return
        self.input_data = await readData(self.input_path)

    #gets the input_data out of the files

    async def start(self):
        await self.read_data()
        await self.prepare_data()


class Controller(Subscriber):
    def __init__(self, name):
        super().__init__(name)

        # init tk
        self.root = tk.Tk()

        #init window size
        self.root.geometry("550x650+200+200")
        self.root.resizable(0, 0)
        #counts running threads
        self.runningAsync = 0

        #init model and viewer
        #init model and viewer with publisher
        self.model = Model(['data_changed', 'clear_data'])
        self.view = View(self.root, self.model, ['start_button', 'close_button'], 'viewer')

        #init Observer
        self.view.subscribe('start_button', self) # Achtung, sich selbst angeben und nicht self.controller
        self.view.subscribe('close_button', self)

    def update(self, event, message):
        print(str(message) + str(event))
        if event == 'start_button':
            self.start_procedure(event)

    def run(self):
        self.root.title("show plot")
        #sets the window in focus
        self.root.deiconify()
        self.root.mainloop()

    def start_procedure(self, event):
        #todo model should notify viewer,
        self.view.hide_instance_attribute(self.view.canvas.get_tk_widget(), 'self.canvas.get_tk_widget()')
        try:
            self.model.input_path = self.view.main.input_path.get()
        except FileNotFoundError:
            messagebox.showerror( 'Error', 'no input_path path')
            return

        try:
            self.model.output_path = self.view.main.output_path.get()
        except FileNotFoundError:
            messagebox.showerror( 'Error', 'no output path')
            return

        try:
            self.model.clearData()
        except ValueError:
            messagebox.showerror('Error', 'could not clear data, restart program')
        self.do_tasks()

        self.view.update_plot()


    def closeprogram(self, event):
        self.root.destroy()

    def closeprogrammenu(self):
        self.root.destroy()

    def do_tasks(self):
        """ Function/Button starting the asyncio part. """
        threading.Thread(target= self.async_load_data, args=()).start()
        print(self.runningAsync)
        while self.model.datadict is None:
            time.sleep(2)
            print('wait')

        # can not use matplotlib outside of the main thread...
        # create plot
        print("ready")
        self.model.fig, self.model.ax = createplot(self.model.datadict['x'], self.model.datadict['y'], self.model.datadict, self.model.output_path)
        self.pca, self.model.fig, self.model.ax = analyze_data(self.model.dataarray, self.model.fig, self.model.ax)

    def async_load_data(self):
        loop = asyncio.new_event_loop()
        self.runningAsync = self.runningAsync + 1
        print(self.runningAsync)
        loop.run_until_complete(self.model.start())
        loop.close()
        self.runningAsync = self.runningAsync - 1


class View(Publisher, Subscriber):
    def __init__(self, parent, model, events, name):
        Publisher.__init__(self, events)
        Subscriber.__init__(self, name)

        #init viewer
        self.model = model
        self.plt = Figure(figsize=(4, 4), dpi=100)
        self.plt.add_subplot(111).plot([0,1, 2, 3, 4],[0,1,20,3,50])
        self.frame = tk.Frame(parent)
        self.frame.grid(sticky="NSEW")
        self.main = Main(parent)
        self.canvas = FigureCanvasTkAgg(self.plt, master=self.frame)
        self.canvas.get_tk_widget().grid(row = 3, column = 0, sticky = tk.N, pady = 2, columnspan = 4)
        self.canvas.draw()

        #init Observer
        self.model.subscribe('data_changed', self) # Achtung, sich selbst angeben und nicht self.controller
        self.model.subscribe('clear_data', self)

        # hidden and shown widgets
        self.hiddenwidgets = {}

        self.main.mainStartButton.bind("<Button>", self.start)
        self.main.quitButton.bind("<Button>", self.closeprogram)

    def hide_instance_attribute(self, instance_attribute, widget_variablename):
        print(instance_attribute)
        self.hiddenwidgets[widget_variablename] = instance_attribute.grid_info()
        instance_attribute.grid_remove()

    def show_instance_attribute(self, widget_variablename):
        try:
            # gets the information stored in
            widget_grid_information = self.hiddenwidgets[widget_variablename]
            print(widget_grid_information)
            # gets variable and sets grid
            eval(widget_variablename).grid(row=widget_grid_information['row'], column=widget_grid_information['column'],
                                           sticky=widget_grid_information['sticky'],
                                           pady=widget_grid_information['pady'],
                                           columnspan=widget_grid_information['columnspan'])
        except:
            messagebox.showerror('Error show_instance_attribute', 'contact developer')

    def start(self, event):
        self.notify("start_button", "start button clicked! Notify subscriber!")

    def closeprogram(self, event):
        self.notify("close_button", "quit button clicked! Notify subscriber!")

    def closeprogrammenu(self):
        self.notify("close_button", "quit button clicked! Notify subscriber!")

    def update_plot(self):
        #todo am besten eine funktion starten, die diese infors kriegt und dann im view ändert
        self.canvas = FigureCanvasTkAgg(self.model.fig, master=self.frame)
        self.show_instance_attribute('self.canvas.get_tk_widget()')


class Main(tk.Frame):
    def __init__(self, root, **kw):

        super().__init__(**kw)
        self.mainFrame = tk.Frame(root)
        self.mainFrame.grid(sticky="NSEW")

        #textfield
        self.input = tk.Label(self.mainFrame, text="Enter input_path path ")
        self.input.grid(row = 0, column = 0, sticky = tk.N, pady = 2, columnspan = 4)

        #entry
        self.input_path = tk.Entry(self.mainFrame, width=80)
        self.input_path.insert(0, 'E:/OneDrive/1_Daten_Dokumente_Backup/1_Laptop_Backup_PC/Programmieren_Python/algorithmn/Algorithmen/PCA/input_path.txt')
        self.input_path.grid(row = 1, column = 0, sticky = tk.N, pady = 2, columnspan = 4)

        #textfield
        self.output = tk.Label(self.mainFrame, text="Enter outputpath")
        self.output.grid(row = 2, column = 0, sticky = tk.N, pady = 2, columnspan = 4)

        #entry
        self.output_path = tk.Entry(self.mainFrame, width=80)
        self.output_path.insert(0,'E:/OneDrive/1_Daten_Dokumente_Backup/1_Laptop_Backup_PC/Programmieren_Python/algorithmn/Algorithmen/PCA/')
        self.output_path.grid(row = 3, column = 0, sticky = tk.N, pady = 2, columnspan = 4)

        #button quit
        self.quitButton = tk.Button(self.mainFrame, text="Quit", width=30, borderwidth=5, bg='#FBD975')
        self.quitButton.grid(row = 7, column = 2, sticky = tk.N, pady = 0)

        #button start
        self.mainStartButton = tk.Button(self.mainFrame, text="Start", width=30, borderwidth=5, bg='#FBD975')
        self.mainStartButton.grid(row = 7, column = 1, sticky = tk.N, pady = 0)
