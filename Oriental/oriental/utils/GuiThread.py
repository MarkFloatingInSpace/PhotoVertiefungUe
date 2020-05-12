# -*- coding: cp1252 -*-


import threading as _threading
import queue as _queue
from sys import exc_info as _exc_info
from .. import config as _config

from matplotlib import pyplot as _pyplot

if _config.gui == "Tk".lower():
    from Tkinter import *

class GuiThread(_threading.Thread):

    def __init__(self,queueArg,queueRes,queueExc):
        _threading.Thread.__init__(self)
        _threading.Thread.daemon = True
        self.queueArg = queueArg
        self.queueRes = queueRes
        self.queueExc = queueExc
        self.isTk = _config.gui == "Tk".lower()
  
    def cb(self):
        #print("cb")
        try:
            # block for 0.05[s] - not too long, or otherwise, the GUI will freeze
            #func, args, kwargs = self.queueArg.get(block=False, timeout=.005)
            func, args, kwargs = self.queueArg.get(block=False)
        except _queue.Empty:
            pass
        else:
            #print func, args, kwargs
            try:
                res = func(*args,**kwargs)
                #print("func called")
            except:
                # the callback must not raise an exception back into the backend's mainloop!
                self.queueArg.task_done()
                self.queueExc.put( _exc_info() )
                if self.isTk:
                    self.root.quit()
            else:
                if self.isTk and len(_pyplot.get_fignums()):
                    fig = _pyplot.gcf()
                    win = fig.canvas.manager.window
                    #if isinstance( win, Tk ):
                    win.focus_force()
                
                self.queueRes.put(res)
                # run the event-loop for a short time
                _pyplot.pause(0.005);
                self.queueArg.task_done()
        
        if self.isTk:
            self.root.after(500,self.cb)
  
    def run(self):
        if self.isTk:
            self.root=Tk()
            self.root.withdraw() # don't display the root window
            self.cb()
            self.root.mainloop()
        else:
            fig = _pyplot.figure()
            timer = fig.canvas.new_timer(interval=5)
            timer.add_callback(self.cb)
            timer.start()
            _pyplot.show()

queueArg = _queue.Queue()
queueRes = _queue.Queue()
queueExc = _queue.Queue()
app = GuiThread( queueArg, queueRes, queueExc )
app.start()


def cmd(func, *args, **kwargs ):
    if app is None  or not app.is_alive():
        raise Exception("oriental.GuiThread is not alive")
    queueArg.put((func, args, kwargs))
    queueArg.join()
    try:
        ex_info = queueExc.get(block=False)
    except _queue.Empty:
        res = queueRes.get()
        queueRes.task_done()
        return res
    else:
        queueExc.task_done()
        # not Python 3 compatible:
        #raise ex_info[0], ex_info[1], ex_info[2]
        raise ex_info[1]