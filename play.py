import tkinter as tk
from tkinter import *
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
import json
import sys
import os
from typing import NoReturn
from PIL import Image,ImageTk
from PIL.ImageTk import PhotoImage
import numpy as np
from tqdm import tqdm
import cv2
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
import threading
import time
import pygame
import pyaudio
import wave
import simpleaudio as sa

WIDTH, HEIGHT = 640, 360
VID_LEN, FPS = 20, 24
K = 1  # search parameter for search area
B_SIZE = 20  # size of macroblocks


def read_image_RGB(fp: str) -> np.ndarray:
    binary_file = np.fromfile(fp, dtype='uint8')
    N = WIDTH * HEIGHT  # num of pixels in the image

    # store rgbs values into a 3-D array to be used for Image.fromarray()
    rgbs = np.zeros((HEIGHT, WIDTH, 3), dtype='uint8')
    for i in range(3):
        rgbs[:, :, i] = np.frombuffer(binary_file[i * N:(i + 1) * N],
                                      dtype='uint8').reshape((-1, WIDTH))
    return rgbs


def scene_detect(name: str, threshold: float = 30.0) -> int:
    video_manager = VideoManager([f'output_video\\{name}.avi'])

    # Warning: windows system use "\"; change to "/" if using linux or macos
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    base_timecode = video_manager.get_base_timecode()
    video_manager.set_downscale_factor()
    print(f'\nDetecting scenes in video "{name}"...')
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)
    return len(scene_list) - 1


class VideoQuery:

    def __init__(self, fp: str):
        name = fp.split(os.sep)[-1]  # extract video name from file path
        category = name.split('_')[0]  # extract category from file path
        self.name = name
        self.listname = name
        self.category = category
        self.fp = fp
        self.pause = False
        # file paths to all frames
        dirs = sorted(os.listdir(self.fp), key=lambda x: int(x[5:-4]))
        fpaths = [f'{self.fp}\\{frame}' for frame in dirs[:VID_LEN * FPS]]
        # read in all frames' rgb values
        print(f'\nReading RGB values of video "{self.name}"...')
        self.data = np.array([read_image_RGB(fp) for fp in tqdm(fpaths)])

    def calc_motion(self) -> int:
        # recast datatype to avoid over/underflow
        self.data = self.data.astype('int64')
        total_motion = 0
        print(f'Calculating motion of video "{self.name}"...')
        with tqdm(total=VID_LEN * FPS - 1) as bar:
            for frame_idx in range(VID_LEN * FPS - 1):
                for y in range(0, HEIGHT, B_SIZE):
                    for x in range(0, WIDTH, B_SIZE):
                        # for each block, find the search area with k
                        start_x, start_y = max(0, x - K), max(0, y - K)
                        end_x, end_y = min(WIDTH - B_SIZE, x + K), \
                                       min(HEIGHT - B_SIZE, y + K)
                        # calculate SAD for each target macro-block and compare
                        # it with the current macro-block to find the best match
                        min_SAD = np.inf
                        best_match = (0, 0)
                        for i in range(start_y, end_y + 1):
                            for j in range(start_x, end_x + 1):
                                sad = self.calc_SAD(frame_idx, i, j, y, x)
                                if sad < min_SAD:
                                    min_SAD = sad
                                    best_match = (i, j)
                        if best_match != (y, x):
                            total_motion += 1
                bar.update(1)
        return total_motion

    def calc_SAD(self, frame_idx: int,
                 c_y: int, c_x: int, n_y: int, n_x: int) -> np.ndarray:
        """
        :param frame_idx: frame index
        :param c_y: current frame Y-coordinate
        :param c_x: current frame X-coordinate
        :param n_y: next frame Y-coordinate
        :param n_x: next frame X-coordinate
        :return: The SAD of Y-values of the macro-block and its co-located
            macro-block in the next frame
        """
        curr_RGB = self.data[frame_idx, c_y:c_y + B_SIZE, c_x:c_x + B_SIZE]
        # curr_YUV = np.apply_along_axis(RGB_to_YUV, 2, curr_RGB)
        next_RGB = self.data[frame_idx + 1, n_y:n_y + B_SIZE, n_x:n_x + B_SIZE]
        # next_YUV = np.apply_along_axis(RGB_to_YUV, 2, next_RGB)
        diff = next_RGB - curr_RGB
        # diff = next_YUV - curr_YUV
        return np.sum(np.abs(diff))

    def to_video(self) -> NoReturn:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        vid_writer = cv2.VideoWriter(f'output_video/{self.name}.avi', fourcc,
                                     FPS, (WIDTH, HEIGHT))
        print(f'Converting "{self.name}" to .avi videos...')
        for frame in self.data:
            vid_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return




class videoGUI:
    def __init__(self, window, window_title,dictionaryname,query_name):
        self.name = query_name.split(os.sep)[-1]  # extract video name from file path
        self.category = self.name.split('_')[0]  # extract category from file path
        self.index = 0
        self.moviePath = f'output_video/{self.name}.avi'
        self.audioPath = f'output_video/{self.category}/{self.name}.wav'
        #music
        self.playMusicQ = False
        self.playMusicL = False
        self.querypath_wav = ''
        self.listpath_wav = ''
        self.indexForQueryMusic = 0

        self.objectQ = ''
        self.objectL = ''

        print(self.audioPath)
        self.listath = self.moviePath
        self.window = window
        self.window.title(window_title)
        self.window.geometry('1920x1080')
        self.window['bg'] = '#333333'
        self.theTop5ListForAllQuery = dictionaryname
        self.frame1 = Frame(window, height=360, width=640)
        self.frame1.pack()
        self.frame1.place(x = 250, y = 460)
        self.canvas = Canvas(self.frame1,height=360, width=640)
        self.canvas.pack()


        self.frame2 = Frame(window, height=360, width=640)
        self.frame2.pack()
        self.frame2.place(x = 1000, y = 460)
        self.canvas2 = Canvas(self.frame2, height=360, width=640)
        self.canvas2.pack()

        # self.im = Image.open("output_video/test/ads/ads_1/ads_1_ads_0_descriptor.png")
        # self.img = ImageTk.PhotoImage(self.im)
        self.frame3 = Frame(window, height=360, width=640)
        self.frame3.pack()
        self.frame3.place(x=1000, y=10)
        self.canvas3 = Canvas(self.frame3, height=360, width=640)
        # self.canvas3.create_image(300,50,image=self.img)
        self.canvas3.pack()



        #this two listname are query and top5 video you select
        self.listname = 'hh'
        self.listname2 = 'ww'
        # pygame.mixer.init()
        # pygame.mixer.music.load(self.audioPath)
        # pygame.mixer.music.play(1,5.0)

        bottom_frame = tk.LabelFrame(self.window)
        bottom_frame.pack(side=BOTTOM, pady=0)

        bottom_frame2 = tk.LabelFrame(self.window)
        bottom_frame2.pack(side=BOTTOM, pady=50)

        # Select Button

        self.btn_select = Button(bottom_frame, text="Loading Query", width=15, command = self.open_file)
        self.btn_select.grid(row=0, column=0)

            # Play Button
        self.btn_play = Button(bottom_frame, text="Play Query audio", width=15,command = self.play_video)
        self.btn_play.grid(row=0, column=1)

            # Pause Button
        self.btn_pause = Button(bottom_frame, text="Pause", width=15, command = self.pause_video)
        self.btn_pause.grid(row=0, column=2)

            # Resume Button
        self.btn_resume = Button(bottom_frame, text="resume", width=15,command = self.resume_video)
        self.btn_resume.grid(row=0, column=3)

        # Select Button 2
        self.btn_select2 = Button(bottom_frame2, text="Loading video", width=15, command = self.open_list_file)
        self.btn_select2.grid(row=0, column=0)

        # Play Button
        self.btn_play2 = Button(bottom_frame2, text="Play Selected matched video", width=15,command = self.play_video2)
        self.btn_play2.grid(row=0, column=1)
        # Pause Button
        self.btn_pause2 = Button(bottom_frame2, text="Pause", command = self.pause_video2)
        self.btn_pause2.grid(row=0, column=2)

        # Resume Button
        self.btn_resume2 = Button(bottom_frame2, text="resume", width=15,command = self.resume_video2)
        self.btn_resume2.grid(row=0, column=3)

        self.var1 = tk.StringVar()
        self.var2 = tk.StringVar()
        self.l1 = tk.Label(self.window, bg='yellow', width=24, height=4, textvariable=self.var1)
        self.l1.pack()
        self.l1.place(x=210)

        self.l2 = tk.Label(self.window, bg='green', width=24, height=4, textvariable=self.var2)
        self.l2.pack()
        self.l2.place(x = 610)

        self.b1 = tk.Button(window, text='Top 5 Select', width=15, height=2, command=self.print_selection)
        self.b1.pack()
        self.b1.place(x=210, y = 100)
        self.b2 = tk.Button(window, text='Query Select', width=15, height=2,command=self.print_query_selection)
        self.b2.pack()
        self.b2.place(x=610, y=100)

        self.desButton = tk.Button(window, text='Descripitor', width=15, height=2, command = self.open_des_file)
        self.desButton.pack()
        self.desButton.place(x=1410, y=200)

        self.var3 = tk.StringVar()
        self.var4 = tk.StringVar()
        self.topList = ['ads_0', 'ads_1', 'ads_2', 'ads_3']
        self.queryList = ['ads_1','ads_2','cartoon_1','cartoon_2','concerts_1','interview_1','interview_2','movies_1','movies_2','sport_1']

        self.var3.set((self.topList))
        self.var4.set((self.queryList))
        self.top5List = tk.Listbox(window, bg='yellow', listvariable=self.var3, font=('Times', 15))
        self.top5List.pack()
        self.top5List.place(height=300, width=200)

        self.qList = tk.Listbox(window, bg='green', listvariable=self.var4, font=('Times', 15))
        self.qList.pack()
        self.qList.place(x = 400, height=300, width=200)

        self.delay = 25  # ms

        self.window.mainloop()

        # Get video's information
        # Open the video file


    #top 5 list
    def print_selection(self):

        self.listname = self.top5List.get(self.top5List.curselection())
        self.var1.set('You select ' + self.listname)

    #query list
    def print_query_selection(self):
        self.listname2 = self.qList.get(self.qList.curselection())
        self.var2.set('You select ' + self.listname2)

    def open_des_file(self):
        self.descriptor = f"output_video/test/{self.listname2.split('_')[0]}/{self.listname2}/{self.listname}.png"
        self.im = Image.open(self.descriptor)
        self.img = ImageTk.PhotoImage(self.im)
        self.canvas3.create_image(300,50,image = self.img)

    def open_file(self):
        self.indexForQueryMusic = 0
        self.pause = False
        print(self.listname2)
        names = self.theTop5ListForAllQuery[self.listname2]
        print(names)
        self.top5List.delete(0,END)
        for i in names:
            self.top5List.insert(END,i)

        # Open the video file
        self.type = self.listname2.split('_')[0]
        self.listpath2 = f'output_video/test/{self.type}/{self.listname2}.avi'
        self.querypath_wav = f'output_video/test/{self.type}/{self.listname2}.wav'
        print("query")
        print(self.listpath2)
        self.cap_q = cv2.VideoCapture(self.listpath2)

        self.width = self.cap_q.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap_q.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.canvas.config(width=self.width, height=self.height)


        pygame.mixer.init()
        # self.objectQ = pygame.mixer.music
        # self.objectQ.load(self.querypath_wav)

        pygame.mixer.music.load(self.querypath_wav)

    def open_list_file(self):

        self.pause = False
        # Open the video file
        self.type = self.listname.split('_')[0]
        print(self.type)
        self.listpath = f'output_video/train/{self.type}/{self.listname}.avi'
        self.listpath_wav = f'output_video/train/{self.type}/{self.listname}.wav'
        print(self.listpath_wav)
        print("list")
        print(self.listpath)
        #print(self.listpath)
        self.cap_l = cv2.VideoCapture(self.listpath)

        self.width = self.cap_l.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.canvas2.config(width=self.width, height=self.height)


        pygame.mixer.init()
        # self.objectL = pygame.mixer.music
        # self.objectL.load(self.listpath_wav)

        pygame.mixer.music.load(self.listpath_wav)


    #Get each frame from video
    def get_frame(self):
        try:
            if self.cap_q.isOpened():
                ret_q, frame_q = self.cap_q.read()
                return (ret_q, cv2.cvtColor(frame_q, cv2.COLOR_BGR2RGB))
        except:
                print('File not Found')
                print("Query Problem")

    def get_frame_list(self):
        try:
            if self.cap_l.isOpened():
                ret_l, frame_l = self.cap_l.read()
                return (ret_l, cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB))
        except:
                print('File not Found')
                print('LIST_PROBLEM')






    def play_video(self):
        ret_q, frame_q = self.get_frame()

        if ret_q:

            if self.indexForQueryMusic == 0:
                pygame.mixer.music.play()
                time.sleep(0.04)
                pygame.mixer.music.pause()
                # self.objectQ.play()
                # time.sleep(0.04)
                # self.objectQ.pause()
            else:
                pygame.mixer.music.unpause()
                time.sleep(0.04)
                pygame.mixer.music.pause()
                # self.objectQ.unpause()
                # time.sleep(0.04)
                # self.objectQ.pause()
            self.indexForQueryMusic = self.indexForQueryMusic + 1

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_q))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

        if self.pause:
            self.window.after_cancel(self.after_id)
            #pygame.mixer.music.pause()
            #pygame.mixer.music.pause()
        else:
            self.after_id = self.window.after(25, self.play_video)



    def play_video2(self):
        ret_l, frame_l = self.get_frame_list()

        if ret_l:
            if self.indexForQueryMusic == 0:
                pygame.mixer.music.play()
                time.sleep(0.04)
                pygame.mixer.music.pause()
                # self.objectL.play()
                # time.sleep(0.04)
                # self.objectL.pause()
            else:
                pygame.mixer.music.unpause()
                time.sleep(0.04)
                pygame.mixer.music.pause()
                # self.objectL.unpause()
                # time.sleep(0.04)
                # self.objectL.pause()
            self.indexForQueryMusic = self.indexForQueryMusic + 1
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_l))

            self.canvas2.create_image(0, 0, image=self.photo, anchor=NW)
        if self.pause:
            self.window.after_cancel(self.after_id)

        else:
            self.after_id = self.window.after(self.delay, self.play_video2)


    #button stop and resume
    def pause_video(self):
        if self.pause == False:
            self.pause = True
        else:
            self.pause = False
            self.play_video()

    def pause_video2(self):
        if self.pause == False:
            self.pause = True
        else:
            self.pause = False
            self.play_video2()

    def resume_video(self):

        self.open_file()
        self.get_frame()
        self.play_video()

    def resume_video2(self):
        self.open_list_file()
        self.get_frame()
        self.play_video2()
    #Release the video source when the object is destroyed
    # def __del__(self):
    #     if self.cap.isOpened():
    #         self.cap.release()

if __name__ == '__main__':
    args = sys.argv

    dict1 = {'ads_1':['ads_0', 'movies_3', 'interview_1', 'cartoon_0', 'cartoon_3'],'ads_2':['ads_1', 'cartoon_2', 'concerts_3', 'interview_2', 'sport_5'],'cartoon_1':['sport_1', 'cartoon_0', 'interview_5', 'sport_0', 'sport_2'],
             'cartoon_2':['ads_0', 'ads_1', 'cartoon_1', 'movies_0', 'movies_2'],'concerts_1':['concerts_2', 'cartoon_4', 'movies_1', 'concerts_0', 'interview_5'],'interview_1':['ads_3', 'movies_0', 'cartoon_2', 'interview_0', 'interview_5'],
             'interview_2':['interview_2', 'movies_1', 'interview_0', 'interview_5', 'interview_3'],'movies_1':['movies_1', 'cartoon_1', 'concerts_0', 'interview_0', 'cartoon_3'],'movies_2':['movie_1', 'cartoon_0', 'sport_4', 'interview_4', 'sport_2'],
             'sport_1':['interview_1', 'sport_2', 'sport_1', 'sport_0', 'sport_5']}



    if len(args) > 1:  # if input video specified, process single video
        fq = args[1]
        print(fq)

        vid_name = fq.split('\\')[-1]

        # vq = VideoQuery(fq)
        # vq.to_video()
        gui = videoGUI(Tk(),'Player',dict1,fq)
        # vq.gui()
        #gui = videoGUI(Tk(),'player')

        sc = scene_detect(vid_name, threshold=20)
        sys.exit()
    #fpath is the address of query's rgb data; fpath_wav is the address of wav data
    fpath = f"C:/Users/Tooth/OneDrive/Desktop/USC/2020_fall/csci576/assignment/project/Data_rgb"
    #fpath = r"C:\Users\Tooth\OneDrive\Desktop\USC\2020_fall\csci576\assignment\project\Data_rgb"

    categories = next(os.walk(fpath))[1]

    cat_paths = [os.path.join(fpath, cat) for cat in categories]
    vid_names = [next(os.walk(cat))[1] for cat in cat_paths]
    #print(vid_names)
    # # commented code below used for converting form rgb to .avi video files
    # vid_paths = [[os.path.join(cat_paths[i], v) for v in cat]
    #              for i, cat in enumerate(vid_names)]
    # videos = [[VideoQuery(vid) for vid in cat] for cat in vid_paths]
    # to_vid = [[vid.to_video() for vid in cat] for cat in videos]
    scenes = {categories[i]:
                  {vid_names[i][j]: scene_detect(vid_names[i][j], 25)
                   for j, vid in enumerate(c)}
              for i, c in enumerate(vid_names)}
    scenes = {"feature_name": "scene_cuts", "values": scenes}
    with open('data.json', 'w') as f:
        json.dump(scenes, f, indent=2, sort_keys=True)