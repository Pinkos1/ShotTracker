## 
# \file gui.py
# \brief Simple GUI with video upload button and modern grey styling
# \author Adam Pinkos
# \date 5/7/25

import tkinter as tk
from tkinter import filedialog, messagebox
from video_reader import read_video

##
# \brief Function triggered when "Upload Video" button is clicked.
# Prompts user to select a video file and plays it back at 5x speed.
def upload_video():
    filepath = filedialog.askopenfilename(
        title = "Select a video file",
        filetypes = [("Video Files", "*.mp4 *.mov *.avi *.mkv"), ("All Files", "*.*")]
    )
    if filepath:
        messagebox.showinfo("File Selected", f"You selected:\n{filepath}")
        read_video(filepath, speed_factor = 20)

# Main window setup
window = tk.Tk()
window.title("Upload Basketball Video")
window.configure(bg = "#2e2e2e") 

# Set window size
win_width = 600
win_height = 300

# Get screen dimensions
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Calculate position to center the window
x = (screen_width // 2) - (win_width // 2)
y = (screen_height // 2) - (win_height // 2)

# Apply size and position
window.geometry(f"{win_width}x{win_height}+{x}+{y}")

# Upload button functionality
upload_btn = tk.Button(
    window,
    text = "Upload Video",
    command = upload_video,
    font=("Helvetica", 14, "bold"), 
    bg = "#444444",
    fg = "white",
    activebackground = "#555555",
    activeforeground = "white",
    relief = "flat",
    padx = 12,
    pady = 8
)
upload_btn.pack(pady = 100)

# Run the GUI event loop
window.mainloop()
