# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from Analyzer import create_training_data_grey, create_training_data_rgb, shuffle_training_data
from Analyzer import prepare_training_data_np, normalize_model, train_model
from Analyzer import save_model_training_data, load_normalized_model, load_trained_model, init_gpu
from Analyzer import save_trained_model, save_normalized_model, load_model_training_data
from Analyzer import layer_sizes, dense_layers, conv_layers, batch_size
from Analyzer import prepare_images_to_test, load_trained_model_only

from observer import Publisher, Subscriber
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import asyncio
import threading
import time


class Model(Publisher):
    def __init__(self, events):
        super().__init__(events)
        # init gpu
        self.gpu_session = init_gpu()

        self.input_path = None
        self.output_path = None
        self.input_data = None

        self.training_data_grey = None
        self.training_data_rgb = None
        self.categories = [ 'Cat', 'Dog']

        self.model_grey = None
        self.model_rgb = None

        self.filepath = None
        self.processing = None

        self.predict_data = None
        self.loaded_model = None

        self.pred_model_path = None

    async def start_prediction_routine(self, name):
        await self.restart_session()
        self.predict_data = await prepare_images_to_test(self.filepath)
        self.loaded_model = await load_trained_model_only(self.pred_model_path)
        await self.predict_model()

    async def predict_model(self):
        # for image in self.predict_data:
        #     prediction = self.model_rgb.predict([image])
        #     print(prediction)  # will be a list in a list.
        #     print(CATEGORIES[int(prediction[0][0])])
        for pred in self.predict_data:
            prediction = self.loaded_model.predict([pred[0]])
            print('img: {} is: {}'.format(pred[1], self.categories[int(prediction[0][0])]))
            print()
        # for pred in prediction:
        #     print(self.categories[pred])

    async def restart_session(self):
        self.gpu_session.close()
        self.gpu_session = await init_gpu()
        time.sleep(5)

    def clearData(self):
        self.input_data = None
        self.training_data_grey = None
        self.training_data_rgb = None
        self.categories = None

    def training_data_loaded(self):
        if self.training_data_grey is None\
                and self.training_data_rgb is None:
            return False
        else:
            return True

    async def create_training_data(self):
        if self.input_path is None:
            messagebox.showerror( 'Error', 'no path!')
            return
        self.training_data_grey = None
        await self.set_process("GREY_Model")
        self.training_data_grey = await create_training_data_grey(self.categories, self.input_path)
        self.training_data_rgb = None
        await self.set_process("RGB_Model")
        self.training_data_rgb = await create_training_data_rgb(self.categories, self.input_path)

    async def shuffle_training_data(self):
        if self.training_data_grey is None\
                and self.training_data_rgb is None:
            messagebox.showerror( 'Error', 'no path!')
            return
        self.training_data_grey = await shuffle_training_data(self.training_data_grey)
        self.training_data_rgb = await shuffle_training_data(self.training_data_rgb)

    async def prepare_training_data_np(self):
        if self.training_data_grey is not None\
                and self.training_data_rgb is not None:
            self.training_data_grey, self.training_data_rgb = await prepare_training_data_np(self.training_data_grey, self.training_data_rgb)
        else:
            messagebox.showerror('Error', 'no training data available')

    async def normalize_model(self):
        layer_size = 32
        dense_layer = 0
        conv_layer = 3
        # await self.restart_session()
        # self.model_grey = await normalize_model(self.training_data_grey, layer_size, dense_layer, conv_layer)
        await self.restart_session()
        self.model_rgb = await normalize_model(self.training_data_rgb, layer_size, dense_layer, conv_layer)

    async def train_model(self):
        batch_s = 32
        # NAME = 'grey'
        # await self.restart_session()
        # self.model_grey = await train_model(self.training_data_grey, self.model_grey, batch_s, NAME, 'grey')
        NAME = 'rgb'
        await self.restart_session()
        time.sleep(5)
        self.model_rgb = await train_model(self.training_data_rgb, self.model_rgb, batch_s, NAME, 'rgb')

    async def prepare_training_data(self, name):
        await self.set_process(name)
        await self.create_training_data()
        await self.shuffle_training_data()
        await self.prepare_training_data_np()
        await self.delete_process()

    async def normalize_model_routine(self, name):
        await self.set_process(name)
        await self.normalize_model()
        await self.delete_process()

    async def train_model_routine(self, name):
        await self.set_process(name)
        await self.train_model()
        await self.delete_process()

    async def save_model_training_data(self, name):
        if self.training_data_grey is not None:
            await self.set_process(name)
            try:
                await save_model_training_data(self.training_data_grey, "grey")
            except AttributeError as e:
                print(e)
            try:
                await save_model_training_data(self.training_data_rgb, "rgb")
            except AttributeError as e:
                print(e)
            await self.delete_process()
        else:
            messagebox.showerror('Error', 'no model_grey available')

    async def save_normalized_model(self, name):
        if self.model_rgb is not None:
            await self.set_process(name)
            await save_normalized_model(self.model_grey, 'grey')
            await save_normalized_model(self.model_rgb, 'rgb')
            await self.delete_process()
        else:
            messagebox.showerror('Error', 'no model_grey available')

    async def save_trained_model(self, name):
        if self.model_rgb is not None:
            await self.set_process(name)
            await save_trained_model(self.model_grey, 'grey')
            await save_trained_model(self.model_rgb, 'rgb')
            await self.delete_process()
        else:
            messagebox.showerror('Error', 'no model_grey available')

    async def load_model_training_data(self, name):
        self.training_data_grey = None
        await self.set_process(name + "_grey")
        self.training_data_grey = await load_model_training_data("grey")
        await self.delete_process()
        self.training_data_rgb = None
        await self.set_process(name + "_rgb")
        self.training_data_rgb = await load_model_training_data("rgb")
        await self.delete_process()

    async def load_normalized_model(self, name):
        self.model_grey, self.training_data_grey = None, None
        await self.set_process(name + "_grey")
        self.model_grey, self.training_data_grey = await load_normalized_model("grey")
        await self.delete_process()

        self.model_rgb, self.training_data_rgb = None, None
        await self.set_process(name + "_rgb")
        self.model_rgb, self.training_data_rgb = await load_normalized_model("rgb")
        await self.delete_process()

    async def load_trained_model(self, name):
        self.model_grey = None
        await self.set_process(name + "_grey")
        self.model_grey, self.training_data_grey = await load_trained_model("grey")
        await self.delete_process()
        self.model_rgb = None
        await self.set_process(name + "_rgb")
        self.model_rgb, self.training_data_rgb = await load_trained_model("rgb")
        await self.delete_process()

    async def set_process(self, task):
        self.dispatch("data_changed", "{} started".format(task))
        self.processing = task

    async def delete_process(self):
        self.dispatch("data_changed", "{} finished".format(self.processing))
        self.processing = None

    async def multiple_model_testing(self, name):
        # await self.set_process(name + "_grey")
        # for dense_layer in dense_layers:
        #     for layer_size in layer_sizes:
        #         for conv_layer in conv_layers:
        #             for batches in batch_size:
        #                 NAME = "{}-conv-{}-nodes-{}-dense-{}-batch-{}_Grey".format(conv_layer, layer_size, dense_layer, batches, int(time.time()))
        #                 print(NAME)
        #                 await self.restart_session()
        #                 self.model_grey = await normalize_model(self.training_data_grey, layer_size, dense_layer, conv_layer)
        #                 await self.restart_session()
        #                 self.model_grey = await train_model(self.training_data_grey, self.model_grey, batches, NAME, 'grey')

        await self.set_process(name + "_rgb")
        for dense_layer in dense_layers:
            for layer_size in layer_sizes:
                for conv_layer in conv_layers:
                    for batches in batch_size:
                        NAME = "{}-conv-{}-nodes-{}-dense-{}-batch-{}_RGB".format(conv_layer, layer_size, dense_layer, batches, int(time.time()))
                        print(NAME)
                        await self.restart_session()
                        self.model_rgb = await normalize_model(self.training_data_rgb, layer_size, dense_layer, conv_layer)
                        await self.restart_session()
                        self.model_rgb = await train_model(self.training_data_rgb, self.model_rgb, batches, NAME, 'rgb')
        await self.delete_process()

class Controller(Subscriber):
    def __init__(self, name):
        super().__init__(name)

        # init tk
        self.root = tk.Tk()

        #init window size
        self.root.geometry("485x750+200+200")
        self.root.resizable(0, 0)
        #counts running threads
        self.runningAsync = 0

        #init model_grey and viewer
        #init model_grey and viewer with publisher
        self.model = Model(['data_changed', 'clear_data'])
        self.view = View(self.root, self.model, ['prepare_training_data',
                                                 'save_model_training_data',
                                                 'save_normalized_model',
                                                 'save_trained_model',
                                                 'train_model_routine',
                                                 'normalize_model_routine',
                                                 'load_model_training_data',
                                                 'load_normalized_model',
                                                 'load_trained_model',
                                                 'multiple_model_testing',
                                                 'start_prediction_routine',
                                                 'close_button'], 'viewer')

        #init Observer
        self.view.register('prepare_training_data', self)  # Achtung, sich selbst angeben und nicht self.controller
        self.view.register('normalize_model_routine', self)
        self.view.register('train_model_routine', self)

        self.view.register('save_model_training_data', self)  # Achtung, sich selbst angeben und nicht self.controller
        self.view.register('save_normalized_model', self)  # Achtung, sich selbst angeben und nicht self.controller
        self.view.register('save_trained_model', self)  # Achtung, sich selbst angeben und nicht self.controller

        self.view.register('load_model_training_data', self)
        self.view.register('load_normalized_model', self)
        self.view.register('load_trained_model', self)

        self.view.register('multiple_model_testing', self)
        self.view.register('start_prediction_routine', self)
        self.view.register('close_button', self)

        #init Observer
        self.model.register('data_changed', self.view) # Achtung, sich selbst angeben und nicht self.controller
        self.model.register('clear_data', self.view)

    def update(self, event, message):
        self.view.write_gui_log("{} button clicked...".format(event))
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
        elif event == 'start_prediction_routine':
            try:
                self.model.filepath = self.view.main.input_image_dir_path.get()
            except FileNotFoundError:
                messagebox.showerror('Error', 'no input path')
                return
            try:
                self.model.pred_model_path = self.view.main.input_model_path.get()
            except FileNotFoundError:
                messagebox.showerror('Error', 'no output path')
                return

        if event == 'close_button':
            self.closeprogram(event)

        self.do_tasks(event)

        self.view.write_gui_log("{} routine finished".format(event))

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
        # self.model_grey -> model_grey
        # self       -> controller
        visit_task = getattr(self.model, task, self.generic_task)
        return await visit_task(task)

    async def generic_task(self, name):
        raise Exception('No model_grey.{} method'.format(name))


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

        # hidden and shown widgets
        self.hiddenwidgets = {}

        self.main.create_training_data_button.bind("<Button>", self.prepare_training_data)
        self.main.normalize_model_button.bind("<Button>", self.normalize_model)
        self.main.train_model_button.bind("<Button>", self.train_model_routine)
        self.main.multiple_model_testing_button.bind("<Button>", self.multiple_model_testing)
        self.main.start_prediction_button.bind("<Button>", self.start_prediction_routine)

        self.main.save_training_data_button.bind("<Button>", self.save_model_training_data)
        self.main.save_normalized_model_button.bind("<Button>", self.save_normalized_model)
        self.main.save_trained_model_button.bind("<Button>", self.save_trained_model)


        self.main.load_training_data_button.bind("<Button>", self.load_model_training_data)
        self.main.load_normalized_model_button.bind("<Button>", self.load_normalized_model)
        self.main.load_trained_model_button.bind("<Button>", self.load_trained_model)

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

    def train_model_routine(self, event):
        self.dispatch("train_model_routine", "train_model_routines clicked! Notify subscriber!")

    def normalize_model(self, event):
        self.dispatch("normalize_model_routine", "normalize_model clicked! Notify subscriber!")

    def multiple_model_testing(self, event):
        self.dispatch("multiple_model_testing", "multiple_model_testing clicked! Notify subscriber!")

    def start_prediction_routine(self, event):
        self.dispatch("start_prediction_routine", "start_prediction_routine clicked! Notify subscriber!")

    def save_model_training_data(self, event):
        self.dispatch("save_model_training_data", "save_model clicked! Notify subscriber!")

    def save_normalized_model(self, event):
        self.dispatch("save_normalized_model", "save_normalized_model clicked! Notify subscriber!")

    def save_trained_model(self, event):
        self.dispatch("save_trained_model", "save_trained_model clicked! Notify subscriber!")

    def load_model_training_data(self, event):
        self.dispatch("load_model_training_data", "load_model_training_data clicked! Notify subscriber!")

    def load_normalized_model(self, event):
        self.dispatch("load_normalized_model", "load_normalized_model clicked! Notify subscriber!")

    def load_trained_model(self, event):
        self.dispatch("load_trained_model", "load_trained_model clicked! Notify subscriber!")

    def closeprogram(self, event):
        self.dispatch("close_button", "quit button clicked! Notify subscriber!")

    def closeprogrammenu(self):
        self.dispatch("close_button", "quit button clicked! Notify subscriber!")

    def update(self, event, message):
        if event == "data_changed":
            self.write_gui_log("{}".format(message))

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

        #button create_training_data
        self.create_training_data_button = tk.Button(self.mainFrame, text="Create Training Data", width=30, borderwidth=5, bg='#FBD975')
        self.create_training_data_button.grid(row = 7, column = 1, sticky = tk.N, pady = 0)

        #button save model_grey
        self.save_training_data_button = tk.Button(self.mainFrame, text="Save Training Data", width=30, borderwidth=5, bg='#FBD975')
        self.save_training_data_button.grid(row = 7, column = 2, sticky = tk.N, pady = 0)

        #button load model_grey
        self.load_training_data_button = tk.Button(self.mainFrame, text="Load Training Data", width=30, borderwidth=5, bg='#FBD975')
        self.load_training_data_button.grid(row = 8, column = 2, sticky = tk.N, pady = 0)

        #button normalize model_grey
        self.normalize_model_button = tk.Button(self.mainFrame, text="Normalize Model", width=30, borderwidth=5, bg='#FBD975')
        self.normalize_model_button.grid(row = 9, column = 1, sticky = tk.N, pady = 0)

        #button  multiple_model_testing_button
        self.multiple_model_testing_button = tk.Button(self.mainFrame, text="multiple_model_testing", width=30, borderwidth=5, bg='#FBD975')
        self.multiple_model_testing_button.grid(row = 13, column = 1, sticky = tk.N, pady = 0)

        #button save norm model_grey
        self.save_normalized_model_button = tk.Button(self.mainFrame, text="Save Normalize Model", width=30, borderwidth=5, bg='#FBD975')
        self.save_normalized_model_button.grid(row = 9, column = 2, sticky = tk.N, pady = 0)

        #button save norm model_grey
        self.load_normalized_model_button = tk.Button(self.mainFrame, text="Load Normalize Model", width=30, borderwidth=5, bg='#FBD975')
        self.load_normalized_model_button.grid(row = 10, column = 2, sticky = tk.N, pady = 0)

        #button train model_grey
        self.train_model_button = tk.Button(self.mainFrame, text="Train Model", width=30, borderwidth=5, bg='#FBD975')
        self.train_model_button.grid(row = 11, column = 1, sticky = tk.N, pady = 0)

        #button save trained model_grey
        self.save_trained_model_button = tk.Button(self.mainFrame, text="Save Trained Model", width=30, borderwidth=5, bg='#FBD975')
        self.save_trained_model_button.grid(row = 11, column = 2, sticky = tk.N, pady = 0)

        #button save trained model_grey
        self.load_trained_model_button = tk.Button(self.mainFrame, text="Load Trained Model", width=30, borderwidth=5, bg='#FBD975')
        self.load_trained_model_button.grid(row = 12, column = 2, sticky = tk.N, pady = 0)

        #button quit
        self.quitButton = tk.Button(self.mainFrame, text="Quit", width=30, borderwidth=5, bg='#FBD975')
        self.quitButton.grid(row = 15, column = 2, sticky = tk.N, pady = 0)

        #### test model ####

        #textfield
        self.input_model = tk.Label(self.mainFrame, text="Enter input path to model ")
        self.input_model.grid(row = 16, column = 0, sticky = tk.N, pady = 2, columnspan = 4)

        #entry
        self.input_model_path = tk.Entry(self.mainFrame, width=80)
        self.input_model_path.insert(0, 'rgb_trained_model_d.h5')
        self.input_model_path.grid(row = 17, column = 0, sticky = tk.N, pady = 2, columnspan = 4)

        #textfield
        self.input_image_dir = tk.Label(self.mainFrame, text="Enter input path ")
        self.input_image_dir.grid(row = 18, column = 0, sticky = tk.N, pady = 2, columnspan = 4)

        #entry
        self.input_image_dir_path = tk.Entry(self.mainFrame, width=80)
        self.input_image_dir_path.insert(0, 'predict/')
        self.input_image_dir_path.grid(row = 19, column = 0, sticky = tk.N, pady = 2, columnspan = 4)

        #button save norm model_grey
        self.start_prediction_button = tk.Button(self.mainFrame, text="start_prediction_button", width=30, borderwidth=5, bg='#FBD975')
        self.start_prediction_button.grid(row = 20, column = 1, sticky = tk.N, pady = 0)


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