"""main file of whiteboard"""

import tkinter as tk
from PIL import Image, ImageDraw
import cv2
import numpy as np
import keras

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CANVAS_WIDTH = 900
CANVAS_HEIGHT = 300
#CENTER = CANVAS_HEIGHT // 2

word_dictionary = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
    9: "I",
    10: "J",
    11: "K",
    12: "L",
    13: "M",
    14: "N",
    15: "O",
    16: "P",
    17: "Q",
    18: "R",
    19: "S",
    20: "T",
    21: "U",
    22: "V",
    23: "W",
    24: "X",
    25: "Y",
    26: "Z",
    27: "*",
}

app = tk.Tk()

tk_word = tk.StringVar()


def main() -> None:
    """whiteboard start"""
    app.title("Pizarra en Blanco")
    app.geometry("1050x570")  # 1050pxx570px
    app.configure(background="#E3ECF2")
    app.resizable(False, False)

    cursor_x = 0
    cursor_y = 0
    color = "#000000"
    width = 10

    def get_cursor_position(work):
        global cursor_x, cursor_y

        cursor_x = work.x
        cursor_y = work.y

    def draw_line1(work):
        global cursor_x, cursor_y

        canvas1.create_line(
            (cursor_x, cursor_y, work.x, work.y),
            width=width,
            fill=color,
            capstyle="round",
            smooth=True,
        )

        draw1.line([cursor_x, cursor_y, work.x, work.y], "black", width=width)

        cursor_x = work.x
        cursor_y = work.y

    # Add icon to app
    logo_icon = tk.PhotoImage(file="word_recognition/assets/logo.png")
    app.iconphoto(False, logo_icon)

    # Declare the canvas
    canvas1 = tk.Canvas(
        app, bg="#ffffff", width=CANVAS_WIDTH, height=CANVAS_HEIGHT, cursor="tcross"
    )
    canvas1.place(x=50, y=10)

    # Declare the canvas to read
    image1 = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
    draw1 = ImageDraw.Draw(image1)

    # Events
    canvas1.bind("<Button-1>", get_cursor_position)
    canvas1.bind("<B1-Motion>", draw_line1)

    # Reset
    def erase_canvas():
        canvas1.delete("all")
        draw1.rectangle((0, 0, CANVAS_WIDTH, CANVAS_HEIGHT), "white")
        image1.save("drawing.jpg")
        global tk_word
        tk_word.set("")

    # Delete Button
    tk.Button(
        app, text="Borrar todo", font=("Arial", 12), bg="#f2f3f5", command=erase_canvas
    ).place(x=50, y=370)

    # Predict!
    def predict():
        my_model = keras.saving.load_model("model.keras", compile=True)

        if my_model is None:
            raise Exception("There is no model saved")

        image1.save("drawing.jpg")
        img = cv2.imread("drawing.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh1 = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV
        )
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
        contours, hierarchy = cv2.findContours(
            dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        im2 = blurred.copy()

        word = ""

        contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Drawing a rectangle on copied image
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #cv2.imshow("Display", rect)

            # Cropping the text block for giving input to OCR
            cropped = cv2.resize(im2[y : y + h, x : x + w], (28, 28))

            cropped = cropped / 255

            print(cropped.dtype)

            cropped = np.fliplr(cropped)
            cropped = np.rot90(cropped, k=1)
            cropped = 1 - cropped

            cropped = np.expand_dims(cropped, axis=0)

            prediction = my_model.predict(cropped)

            word += word_dictionary[np.argmax(prediction)]

        print(word) #::-1
        global tk_word

        tk_word.set(word) #::-1

    tk.Button(
        app,
        text="Predecir",
        font=("Arial", 12),
        bg="#f2f3f5",
        command=predict,
    ).place(x=350, y=370)

    tk.Label(app, textvariable=tk_word, font="Arial, 12", bg="white", fg="black").place(
        x=50, y=470
    )

    app.mainloop()
