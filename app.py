import tkinter as tk
from tkinter import scrolledtext
from test import manual_testing   # on importe ta fonction depuis model.py

app = tk.Tk()
app.title("Fake News Detector")
app.geometry("600x500")
app.config(bg="#F8F8F8")

label = tk.Label(app, text="Écris une news à tester :", font=("Arial", 14), bg="#F8F8F8")
label.pack(pady=10)

text_area = scrolledtext.ScrolledText(app, wrap=tk.WORD, width=70, height=10, font=("Arial", 12))
text_area.pack(pady=10)

result_label = tk.Label(app, text="", font=("Arial", 16, "bold"), bg="#F8F8F8")
result_label.pack(pady=20)

def detect():
    user_text = text_area.get("1.0", tk.END).strip()
    if len(user_text) == 0:
        result_label.config(text="⚠️ Entrez un texte !", fg="red")
    else:
        result = manual_testing(user_text)
        result_label.config(text=result, fg="green")

button = tk.Button(app, text="Tester la news", font=("Arial", 14), command=detect, bg="#3498db", fg="white")
button.pack(pady=10)

app.mainloop()
