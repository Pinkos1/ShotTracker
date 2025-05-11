## 
# \file gui.py
# \brief Simple GUI with video upload button
# \author Adam Pinkos
# \date 5/7/25

import tkinter as tk
from tkinter import filedialog, messagebox
from ShotTracker.video_reader import read_video  # import function from module

##
# \brief Function triggered when "Upload Video" button is clicked.
# Prompts user to select a video file and plays it back at 5x speed.
def upload_video():
    filepath = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv"), ("All Files", "*.*")]
    )
    if filepath:
        messagebox.showinfo("File Selected", f"You selected:\n{filepath}")
        read_video(filepath, speed_factor = 10)  

# main window
window = tk.Tk()
window.title("Upload Basketball Video")
window.geometry("350x150")

# Create the upload button
upload_btn = tk.Button(window, text = "Upload Video", command = upload_video)
upload_btn.pack(pady= 50)

# Run the GUI event loop
window.mainloop()
