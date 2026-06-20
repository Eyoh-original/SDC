from tkinter import *
from PIL import ImageTk, Image
import cv2

class Athersat_UI:
    def __init__(self, stream_source="http://10.236.206.199:5000/video"):
        self.root = Tk()
        self.root.title("Athersat UI")
        self.root.geometry("800x600")

        self.label = Label(self.root)
        self.label.pack()

        self.cap = cv2.VideoCapture(stream_source)

        self.video_stream()

    def video_stream(self):
        if not self.cap.isOpened():
            self.root.after(1000, self.video_stream)
            return

        ret, frame = self.cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.root.after(10, self.video_stream)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = Athersat_UI()
    app.run()
