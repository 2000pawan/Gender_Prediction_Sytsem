import cv2
import joblib
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

model = joblib.load('model.pkl')  # Load your trained gender prediction model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ------------------ LOGIN PAGE ------------------
def login():
    username = entry_user.get()
    password = entry_pass.get()
    if username == "admin" and password == "admin":
        login_frame.destroy()
        show_main_menu()
    else:
        messagebox.showerror("Error", "Invalid Username or Password")

# ------------------ LOGOUT FUNCTION ------------------
def logout():
    # Destroy all widgets and recreate login page
    for widget in win.winfo_children():
        widget.destroy()
    create_login_page()

# ------------------ CREATE LOGIN PAGE ------------------
def create_login_page():
    # Change project name label background to orange
    Label(win, text="Gender Prediction System", font=('', 45, 'bold'), bg='orange').pack(pady=30)

    global login_frame
    login_frame = Frame(win, bg='sky blue')
    login_frame.pack(pady=20)

    Label(login_frame, text='Login', font=('', 40, 'bold'), bg='sky blue').pack(pady=20)

    Label(login_frame, text='Username:', font=('Arial', 20), bg='sky blue').pack()
    global entry_user
    entry_user = Entry(login_frame, font=('Arial', 20))
    entry_user.pack(pady=10)

    Label(login_frame, text='Password:', font=('Arial', 20), bg='sky blue').pack()
    global entry_pass
    entry_pass = Entry(login_frame, font=('Arial', 20), show='*')
    entry_pass.pack(pady=10)

    Button(login_frame, text='Login', command=login, font=('Arial', 20), bg='green', fg='white').pack(pady=20)

    # Footer at bottom
    footer = Label(win, text="Developed by @Pawan Yadav (Ai Engineer)  All rights reserved 2025", 
                   font=('Arial', 20,'bold'), bg='red')
    footer.place(relx=1.0, rely=1.0, anchor='se', x=-20, y=-10)


# ------------------ MAIN MENU ------------------
def show_main_menu():
    # Clear all widgets first
    for widget in win.winfo_children():
        widget.destroy()

    # Create logout button at top-right
    logout_btn = Button(win, text="Logout", command=logout, font=('Arial', 14), bg='red', fg='white')
    logout_btn.pack(anchor='ne', padx=20, pady=10)

    # Change project name label background to orange
    Label(win, text="Gender Prediction System", font=('', 45, 'bold'), bg='orange').pack(pady=10)

    # Frame to hold 3 buttons with space
    btn_frame = Frame(win, bg='sky blue')
    btn_frame.pack(pady=60)

    # Buttons spaced evenly horizontally with padding
    Button(btn_frame, text="Predict from Image", command=predict_image, font=('Arial', 20), bg='lightblue', width=20).grid(row=0, column=0, padx=20)
    Button(btn_frame, text="Predict from Video", command=predict_video, font=('Arial', 20), bg='lightgreen', width=20).grid(row=0, column=1, padx=20)
    Button(btn_frame, text="Predict from Camera", command=predict_camera, font=('Arial', 20), bg='salmon', width=20).grid(row=0, column=2, padx=20)

    # Footer at bottom
    footer = Label(win, text="Developed by @Pawan Yadav (Ai Engineer)  All rights reserved 2025", 
                   font=('Arial', 20,'bold'), bg='red')
    footer.place(relx=1.0, rely=1.0, anchor='se', x=-20, y=-10)


# ------------------ PREDICT FROM IMAGE ------------------
def predict_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path)
    img = cv2.resize(img, (400, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (90, 90)).flatten().reshape(1, -1) / 255.0
        pred = model.predict(face)[0]
        label = f"Gender: {pred}"
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    show_image(img)

def show_image(cv_img):
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    top = Toplevel(win)
    top.title("Result")
    top.resizable(False, False)   # Make popup non-resizable
    lbl = Label(top, image=imgtk)
    lbl.image = imgtk
    lbl.pack()

# ------------------ PREDICT FROM VIDEO ------------------
def predict_video():
    path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if not path:
        return

    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (90, 90)).flatten().reshape(1, -1) / 255.0
            pred = model.predict(face)[0]
            label = f"Gender: {pred}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("Gender Prediction - Video", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ------------------ PREDICT FROM CAMERA ------------------
def predict_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (90, 90)).flatten().reshape(1, -1) / 255.0
            pred = model.predict(face)[0]
            label = f"Gender: {pred}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
       
        cv2.imshow("Gender Prediction - Camera", frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ------------------ MAIN WINDOW ------------------
win = Tk()
win.state('zoomed')
win.configure(bg='sky blue')
win.title('Gender Prediction System')

create_login_page()

win.mainloop()
