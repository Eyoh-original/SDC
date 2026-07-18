from tkinter import *
from PIL import ImageTk, Image
import cv2

class Athersat_UI:
    def __init__(self, stream_source="http://172.19.154.199:5000/video"):
        self.root = Tk()
        self.root.title("Athersat UI")
        self.root.geometry("800x600")

        self.video_frame = Frame(self.root, width=500, height=375)
        self.video_frame.pack(side=LEFT, padx=10, pady=10)

        self.control_frame = Frame(self.root, width=250, height=500)
        self.control_frame.pack(side=RIGHT, padx=10, pady=10)

        self.label = Label(self.video_frame)
        self.label.grid()

        self.right_button = Button(self.control_frame, text="Right", command=self.click)
        self.right_button.grid(row=0, column=1, pady=20, padx=20)

        self.left_button = Button(self.control_frame, text="Left", command=self.click)
        self.left_button.grid(row=0, column=0, pady=20, padx=20)

        self.cap = cv2.VideoCapture(stream_source)

        self.video_stream()

    def video_stream(self):
        if not self.cap.isOpened():
            self.root.after(1000, self.video_stream)
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (500, 375))
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.root.after(10, self.video_stream)

#class PID_GUI:
    #def __init__(self, root):
        #self.root = root
        #self.frame = LabelFrame(root, text="PID Controller", padx=10, pady=10, bg ="lightgray")
        #self.btn.right = Button(self.frame, text="Right", command=self.click)
        #self.btn.left = Button(self.frame, text="Left", command=self.click)
        
        #self.publish()
    
    #def publish(self):
        #self.frame.grid(row=0, column=0, rowspan=3, columnspan=3, padx=10, pady=10)
        #self.btn.right.grid(row=0, column=1, pady=20, padx=20)
        #self.btn.left.grid(row=0, column=0, pady=20, padx=20)

    def click(self):
        print("Right button clicked")

    def run(self):
        self.root.mainloop()




if __name__ == "__main__":
    app = Athersat_UI()
    app.run()
    #PID_GUI(app.root)
