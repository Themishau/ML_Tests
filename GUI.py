# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from Analyzer import create_training_data, shuffle_training_data, create_model, save_model, load_model, createplot, create_img, analyze_data
from observer import Publisher, Subscriber
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import asyncio
import threading
import time

class Model(Publisher):
    def __init__(self, events):
        super().__init__(events)
        self.input_data = None
        self.training_data = None
        self.categories = [ 'Cat', 'Dog']
        self.model = None
        self.processing = None

        self.fig = None
        self.ax = None

        self.input_path = None
        self.output_path = None

    def clearData(self):
        self.input_data = None
        self.training_data = None
        self.categories = None

    def training_data_loaded(self):
        if self.training_data is None:
            return False
        else:
            return True

    async def create_training_data(self):
        if self.input_path is None:
            messagebox.showerror( 'Error', 'no path!')
            return
        self.training_data = None
        self.training_data = await create_training_data(self.categories, self.input_path)
    async def shuffle_training_data(self):
        if self.training_data is None:
            messagebox.showerror( 'Error', 'no path!')
            return
        self.training_data = await shuffle_training_data(self.training_data)
        #gets the input_data out of the files

    async def prepare_training_data(self, name):
        await self.set_process(name)
        await self.create_training_data()
        await self.shuffle_training_data()
        await self.delete_process()

    async def create_model(self, name):
        await self.set_process(name)
        self.model = await create_model(self.training_data)
        await self.delete_process()

    async def save_model(self, name):
        if self.model is not None:
            await self.set_process(name)
            await save_model(self.model)
            await self.delete_process()

    async def load_model(self, name):
        self.model = None
        await self.set_process(name)
        self.model = await load_model()
        await self.delete_process()

    async def set_process(self, task):
        self.processing = task

    async def delete_process(self):
        self.processing = None


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
        self.view = View(self.root, self.model, ['prepare_training_data', 'create_model', 'save_model', 'load_model', 'close_button'], 'viewer')

        #init Observer
        self.view.register('prepare_training_data', self)  # Achtung, sich selbst angeben und nicht self.controller
        self.view.register('create_model', self)  # Achtung, sich selbst angeben und nicht self.controller
        self.view.register('save_model', self)  # Achtung, sich selbst angeben und nicht self.controller
        self.view.register('load_model', self)
        self.view.register('close_button', self)

    def update(self, event, message):
        self.view.write_gui_log("{} start...".format(event))
        if event == 'prepare_training_data':
            try:
                self.model.input_path = self.view.main.input_path.get()
            except FileNotFoundError:
                messagebox.showerror('Error', 'no input path')
                return
            try:
                self.model.output_path = self.view.main.output_path.get()
            except FileNotFoundError:
                messagebox.showerror('Error', 'no output path')
                return
            self.do_tasks(event)

        if event == 'create_model':
            self.do_tasks(event)

        if event == 'save_model':
            self.do_tasks(event)

        if event == 'load_model':
            self.do_tasks(event)

        if event == 'close_button':
            self.closeprogram(event)
        self.view.write_gui_log("{} done".format(event))
    def run(self):
        self.root.title("show plot")
        #sets the window in focus
        self.root.deiconify()
        self.root.mainloop()

    def closeprogram(self, event):
        self.root.destroy()

    def closeprogrammenu(self):
        self.root.destroy()

    def do_tasks(self, task):
        """ Function/Button starting the asyncio part. """
        return threading.Thread(target= self.async_do_task(task), args=()).start()

    def async_do_task(self, task):
        loop = asyncio.new_event_loop()
        self.runningAsync = self.runningAsync + 1
        visit_task = getattr(self.model, task, self.generic_task)
        loop.run_until_complete(visit_task(task))
        while self.model.processing is not None:
            time.sleep(1)
            print('status: {} please wait...'.format(self.model.processing))
        loop.close()
        self.runningAsync = self.runningAsync - 1

    async def task(self, task):

        # create an generic method call
        # self.model -> model
        # self       -> controller
        visit_task = getattr(self.model, task, self.generic_task)
        return await visit_task(task)

    async def generic_task(self, name):
        raise Exception('No model.{} method'.format(name))

class View(Publisher, Subscriber):
    def __init__(self, parent, model, events, name):
        Publisher.__init__(self, events)
        Subscriber.__init__(self, name)

        #init viewer
        self.model = model
        self.sidePanel = InfoBottomPanel(parent)
        self.frame = tk.Frame(parent)
        self.frame.grid(sticky="NSEW")
        self.main = Main(parent)

        #init Observer
        self.model.register('data_changed', self) # Achtung, sich selbst angeben und nicht self.controller
        self.model.register('clear_data', self)

        # hidden and shown widgets
        self.hiddenwidgets = {}

        self.main.create_training_data_button.bind("<Button>", self.prepare_training_data)
        self.main.create_model_button.bind("<Button>", self.create_model)
        self.main.save_model_button.bind("<Button>", self.save_model)
        self.main.load_model_button.bind("<Button>", self.load_model)
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

    # four events:
    # 'create_training_data',
    # 'create_model',
    # 'save_model',
    # 'close_button'
    def prepare_training_data(self, event):
        self.dispatch("prepare_training_data", "prepare_training_data clicked! Notify subscriber!")

    def create_model(self, event):
        self.dispatch("create_model", "create_model clicked! Notify subscriber!")

    def save_model(self, event):
        self.dispatch("save_model", "save_model clicked! Notify subscriber!")

    def load_model(self, event):
        self.dispatch("load_model", "load_model clicked! Notify subscriber!")

    def closeprogram(self, event):
        self.dispatch("close_button", "quit button clicked! Notify subscriber!")

    def closeprogrammenu(self):
        self.dispatch("close_button", "quit button clicked! Notify subscriber!")

    def update_plot(self):
        #todo am besten eine funktion starten, die diese infors kriegt und dann im view ändert
        self.canvas = FigureCanvasTkAgg(self.model.fig, master=self.frame)
        self.show_instance_attribute('self.canvas.get_tk_widget()')

    def write_gui_log(self, text):
        time_now = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
        self.sidePanel.log.insert("end", str(time_now) + ': ' + text)
        self.sidePanel.log.yview("end")

class Main(tk.Frame):
    def __init__(self, root, **kw):

        super().__init__(**kw)
        self.mainFrame = tk.Frame(root)
        self.mainFrame.grid(sticky="NSEW")

        #textfield
        self.input = tk.Label(self.mainFrame, text="Enter input path ")
        self.input.grid(row = 0, column = 0, sticky = tk.N, pady = 2, columnspan = 4)

        #entry
        self.input_path = tk.Entry(self.mainFrame, width=80)
        self.input_path.insert(0, 'Data/')
        self.input_path.grid(row = 1, column = 0, sticky = tk.N, pady = 2, columnspan = 4)

        #textfield
        self.output = tk.Label(self.mainFrame, text="Enter outputpath")
        self.output.grid(row = 2, column = 0, sticky = tk.N, pady = 2, columnspan = 4)

        #entry
        self.output_path = tk.Entry(self.mainFrame, width=80)
        self.output_path.insert(0,'E:/OneDrive/1_Daten_Dokumente_Backup/1_Laptop_Backup_PC/Programmieren_Python/algorithmn/Algorithmen/ML/ML_Tests')
        self.output_path.grid(row = 3, column = 0, sticky = tk.N, pady = 2, columnspan = 4)

        #button quit
        self.quitButton = tk.Button(self.mainFrame, text="Quit", width=30, borderwidth=5, bg='#FBD975')
        self.quitButton.grid(row = 7, column = 2, sticky = tk.N, pady = 0)

        #button create_training_data
        self.create_training_data_button = tk.Button(self.mainFrame, text="Create Training Data", width=30, borderwidth=5, bg='#FBD975')
        self.create_training_data_button.grid(row = 7, column = 1, sticky = tk.N, pady = 0)

        #button create model
        self.create_model_button = tk.Button(self.mainFrame, text="Create Model", width=30, borderwidth=5, bg='#FBD975')
        self.create_model_button.grid(row = 8, column = 1, sticky = tk.N, pady = 0)

        #button save model
        self.save_model_button = tk.Button(self.mainFrame, text="Save Model", width=30, borderwidth=5, bg='#FBD975')
        self.save_model_button.grid(row = 9, column = 1, sticky = tk.N, pady = 0)

        #button save model
        self.load_model_button = tk.Button(self.mainFrame, text="Load Model", width=30, borderwidth=5, bg='#FBD975')
        self.load_model_button.grid(row = 10, column = 1, sticky = tk.N, pady = 0)

class InfoBottomPanel(tk.Frame):
    def __init__(self, root, **kw):
        super().__init__(**kw)
        self.sidepanel_frame = tk.Frame(root)
        self.sidepanel_frame.grid(sticky="NSEW")
        self.entry = tk.Label(self.sidepanel_frame, text="Log")
        self.entry.grid(row=0, column=0, sticky=tk.N, pady=0, columnspan=4)
        self.log = tk.Listbox(self.sidepanel_frame, width=80)
        self.log_scroll = tk.Scrollbar(self.sidepanel_frame, orient="vertical")
        self.log.config(yscrollcommand=self.log_scroll.set)
        self.log_scroll.config(command=self.log.yview)
        self.log.grid(row=1, column=0, sticky=tk.N, pady=0, columnspan=4)