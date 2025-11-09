import tkinter as tk  # Python 3
import time
import os
import tkinter.font as tkFont
from tkinter.constants import BOTH
from tkinter.ttk import Label
from tkinter import Canvas, Tk
import sys
from PIL import ImageTk, Image

root = tk.Tk()

frameCnt = 12
frame = tk.PhotoImage(file='C:\\Users\\a1\\Desktop\\Newfolder\\1.png')

helv36 = tkFont.Font(family='Helvetica', size=14)

lastClickX = 0
lastClickY = 0


def SaveLastClickPos(event):
    global lastClickX, lastClickY
    lastClickX = event.x
    lastClickY = event.y


def Dragging(event):
    x, y = event.x - lastClickX + root.winfo_x(), event.y - lastClickY + \
        root.winfo_y()
    root.geometry("+%s+%s" % (x, y))
    root.attributes('-alpha', 0.3)


def Release(event):
    root.attributes('-alpha', 1)


class FancyListbox(tk.Listbox):

    def __init__(self, parent, *args, **kwargs):
        tk.Listbox.__init__(self, parent, *args, **kwargs)

        self.popup_menu = tk.Menu(self, tearoff=0)
        self.popup_menu.add_command(label="聊天",
                                    command=self.chat)
        self.popup_menu.add_command(label="选项（TBD）",
                                    command=self.options)
        self.popup_menu.add_command(label="关闭",
                                    command=self.close)

        self.bind("<Button-3>", self.popup)  # Button-2 on Aqua
        self.popup_menu.config(font=helv36)

    def popup(self, event):
        try:
            self.popup_menu.tk_popup(event.x_root, event.y_root, 0)
        finally:
            self.popup_menu.grab_release()

    def callback(self, text, master):
        print(text)
        master.destroy()

    def sendCmd(self, cmd):
        print(cmd)

    def chat(self):
        child = tk.Toplevel(self)
        child.transient(self)
        child.title("和她说话吧！")
        child.geometry("480x360")

        # b1 = tk.Button(child, command=self.fun)
        # b1.pack(side=tk.LEFT)

        child.wm_attributes('-transparentcolor', '#ab23ff')

        e = tk.Text(child)
        e.place(x=0, y=0, relwidth=1, relheight=0.8)
        e.config(font=helv36)
        # e.pack()

        child.image = tk.PhotoImage(
            file='C: \\Users\\a1\\Desktop\\Newfolder\\chatbtn.png')
        label = tk.Label(child, image=child.image, bg='white')
        label.pack(side=tk.BOTTOM)

        # self.sendCmd("end-1c")

        x = 'dummy'
        y = 'dummy'
        z = 'dummy'

        label.bind("<Button-1>", lambda x: self.sendCmd(e.get("1.0", "end-1c")))

        self.update_idletasks()
        child.mainloop()

    def options(self):
        # self.selection_set(0, 'end')
        print('options')

    def close(self):
        # self.selection_set(0, 'end')
        print('close')
        root.destroy()


def update(ind):

    label.configure(image=frame)
    root.after(100, update, ind)


# The image must be stored to Tk or it will be garbage collected.
root.image = tk.PhotoImage(file='1.png')
label = tk.Label(root, image=root.image, bg='white')

flb = FancyListbox(root, selectmode='single')
# for n in range(10):
#    flb.insert('end', n)
# flb.pack()

root.overrideredirect(True)
root.geometry("+250+250")
root.lift()
root.wm_attributes("-topmost", True)
# root.wm_attributes("-disabled", True)
root.wm_attributes("-transparentcolor", "white")
root.bind("<Button-3>", flb.popup)
root.bind('<Button-1>', SaveLastClickPos)
root.bind('<B1-Motion>', Dragging)
root.bind('<ButtonRelease>', Release)

label.pack()
root.after(0, update, 0)

root.protocol("WM_DELETE_WINDOW", FancyListbox.close)
label.mainloop()
