import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from neural_network import predict_melanon

class Application:
    #tvorba uzivatelskeho rozhrani
    def __init__(self, master):
        self.pics = []
        self.master = master

        self.canvas = tk.Canvas(master, height = 550, width = 550, bg = "white")
        self.canvas.pack()

        self.frame = tk.Frame(master, bg = "white")
        self.frame.place(relwidth = 0.8, relheight = 0.8, relx = 0.1, rely = 0.1)

        self.openFile = tk.Button(master, text = "Open File", padx=10,
                         pady=5, fg="black", bg="AntiqueWhite2", command = self.loadTestPicture)
        self.openFile_window = self.canvas.create_window(10, 10, anchor="nw", window = self.openFile)

        self.predictMelanom = tk.Button(master, text = "Predict Melanom",
                            padx=10, pady=5, fg="black", bg="AntiqueWhite2", command = self.prediction)
        self.predictMelanom_window = self.canvas.create_window(210, 500, anchor = "nw", window = self.predictMelanom )

        self.quit = tk.Button(master, text = "Quit",
                            padx=10, pady=5, fg="black", bg="AntiqueWhite2", command = self.quit)
        self.quit_window = self.canvas.create_window(490, 10, anchor="nw", window = self.quit)

        self.label = tk.Label(master, text="Please load your test image", bg = "white")
        self.label_window = self.canvas.create_window(10, 475, anchor="nw", window = self.label)

    def get_pics(self):
        return self.pics
    #nacteni obrazku
    def loadTestPicture(self):
        self.pics.clear()
        self.label.config(text="Please load your test image", fg = "black")
        file_path = filedialog.askopenfilename(initialdir="/",title = "Vyberte Obrázek",
                                            filetypes=(("images (*.png, *.jpg)", "*.png"),
                                                       ("images (*.png, *.jpg)", "*.jpg")))
        if not file_path:
            self.label.config(text="Please load your test image!")
        else:
            self.pics.append(file_path)
            print(file_path)
            self.loadImage(self.frame, file_path)
    #zobrazeni obrazku
    def loadImage(self, frame, file_path):
        for widget in frame.winfo_children():
            widget.destroy()

        img = Image.open(file_path)
        img = img.resize((400, 400))
        filename = ImageTk.PhotoImage(img)
        label = tk.Label(frame, image=filename)
        label.image = filename
        label.place(x=5, y=5)
        label.pack()
        self.label.config(text="Image successfully loaded")
    #predikce vysledku
    def prediction(self):
        if not self.pics:
            self.label.config(text="Please load your test image")
            #print("List is empty")
        else:
            verdict = 100 * predict_melanon(self.pics[0])
            if verdict[0][0] < 50:
                self.label.config(fg = "green")
            elif verdict[0][0] > 50 and verdict[0][0] < 75:
                self.label.config(fg = "black")
            else:
                self.label.config(fg = "red")
            verdict = round(verdict[0][0], 2).astype("str")
            self.label.config(text = "Probability of melanom is " + verdict + "%")
            #print(verdict + "%")

    def quit(self):
        self.master.destroy()

window = tk.Tk()
window.title('Riziko Melanomu')
app = Application(window)
window.mainloop()