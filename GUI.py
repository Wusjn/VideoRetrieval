import tkinter as tk
from tkinter import ttk
import pickle
from videoRetrieval import search_video
from PIL import ImageTk
with open("./data/HMDB_search_split.pkl", "rb") as file:
    search_split = pickle.load(file)

search_dict = {}
for entry in search_split:
    type_number, video_type, video_name = entry
    if video_type not in search_dict:
        search_dict[video_type] = []
    search_dict[video_type].append(video_name)

window = tk.Tk()

window.title('Video Retrieval')

window.geometry('800x500')


selected_video = ["brush_hair","Aussie_Brunette_Brushing_Long_Hair_brush_hair_u_nm_np1_ba_med_3.avi"]

def video_type_selected(event):
    global selected_video
    video_name_cmb["value"] = search_dict[video_type_cmb.get()]
    video_name_cmb.current(0)
    selected_video[0] = video_type_cmb.get()
    selected_video[1] = video_name_cmb.get()
    print(selected_video)
video_type_cmb = ttk.Combobox(window)
video_type_cmb.pack()
video_type_cmb['value'] = list(search_dict.keys())
video_type_cmb.current(0)
video_type_cmb.bind("<<ComboboxSelected>>",video_type_selected)


def video_name_selected(event):
    global selected_video
    selected_video[1] = video_name_cmb.get()
video_name_cmb = ttk.Combobox(window)
video_name_cmb.pack()
video_name_cmb['value'] = search_dict["brush_hair"]
video_name_cmb.current(0)
video_name_cmb.bind("<<ComboboxSelected>>",video_name_selected)

def search_video_action():
    t.delete('1.0','end')
    scores = search_video(selected_video)
    for score in scores:
        t.insert("end",score[1][1] + "/" + score[1][2] + "\n")

search_button = tk.Button(window, text='search video', font=('Arial', 12), width=10, height=1, command=search_video_action)
search_button.pack()

t = tk.Text(window, height=10)
t.pack()

window.mainloop()
