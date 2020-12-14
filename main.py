# external modules
import os
from datetime import datetime

from tkinter import TclError

# internal modules
import internal.gui as gui
import internal.yolo as yolo

YOLO_MODEL = yolo.load_model()

def output_filename_format(path):
    return os.path.sep.join([
        os.path.dirname(os.path.abspath(path)),
        datetime.now().strftime('%d-%m-%Y_%H.%M.%S') + '__' + os.path.basename(path)
        ])

def on_file_choose(ui_handle, path):
    # Create new instance of the VideoProcesser with parameters and run it

    vp = yolo.VideoProcesser(ui_handle=ui_handle)
    vp.set_model(YOLO_MODEL)

    output_path = output_filename_format(path)
    vp.open_io(path, output_path)

    vp.run()

    ui_handle.choose_button['state'] = 'normal'
    ui_handle.update_label('Choose video to analyze:')
    ui_handle.update_status('(not currently processing)')


callbacks = {
    'on_file_choose': on_file_choose
}

try:
    app = gui.App(callbacks=callbacks)
    app.run()
except TclError:
    pass
