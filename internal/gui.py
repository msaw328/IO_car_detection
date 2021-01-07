import tkinter as tk

from tkinter import filedialog, messagebox

def show_info(title, message):
    return messagebox.showinfo(title=title, message=message)

def show_error_and_quit(title, message):
    messagebox.showerror(title=title, message=message)
    exit()

class App:
    # handler for button click
    def __choose_handler(self):
        paths = filedialog.askopenfilenames(
            title="Select a file", 
            filetypes=(
                ("avi files", "*.avi"),
                ("mp4 files", "*.mp4")
            )
        )

        paths = list(paths)

        for path in paths:
            # if user cancels, return
            if not path.strip():
                continue

            # disable the button, set different label
            self.choose_button['state'] = 'disabled'
            self.update_label('Currently processing: ' + path)

            # run the callback with path as the only argument
            if self.callbacks and 'on_file_choose' in self.callbacks:
                self.callbacks['on_file_choose'](self, path)

        if len(paths) > 0:
            show_info('Processing complete', 'Processing of all videos has completed')

    # allows updating of the status bar text
    def update_status(self, text):
        self.status_bar['text'] = str(text)
        self.root.update()

    # allows updating of the label text
    def update_label(self, text):
        self.label['text'] = str(text)
        self.root.update()

    def __init__(self, callbacks):
        self.callbacks = callbacks

        self.root = tk.Tk()
        self.root.resizable(False, False)
        self.root.title("Vehicle Detection")
        
        self.label = tk.Label(self.root, text="Choose video to analyze:", font="Helvetica 15 bold", padx=50, pady=25)
        self.label.grid(row=1)

        self.choose_button = tk.Button(self.root, text="Choose file", font="Helvetica 15 bold", padx=20, pady=20, command=self.__choose_handler, bg="yellow")
        self.choose_button.grid(row=2)

        self.status_bar = tk.Label(self.root, text="(not currently processing)", relief="sunken")
        self.status_bar.grid(row=5, sticky="WE")

    def run(self):
        self.root.mainloop()


            
