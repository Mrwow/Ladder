from moviepy.editor import VideoFileClip,concatenate_videoclips
import imageio
import os
import cv2
import numpy as np

def video2gif(video,size=4, n=7):
    gif_name = video.replace(".mov",'.gif')
    video = cv2.VideoCapture(video)
    fps = video.get(5)

    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'fame rate is {fps}, fame number is {frames}')

    # gif = []
    # count = 1
    # while video.isOpened():
    #     ret, frame = video.read()
    #     if ret ==False:
    #         break
    #     img = cv2.resize(frame, (int(frame.shape[1]/size),int(frame.shape[0]/size)))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     if count%n == 0:
    #         gif.append(img)
    #     count += 1
    #
    # imageio.mimsave(gif_name,gif,'GIF', duration=0.1)

if __name__ == '__main__':
    video = "/Users/ZhouTang/Desktop/docImg/detection.mov"
    video2gif(video, size=3,n=15)

    videoClip = VideoFileClip("/Users/ZhouTang/Desktop/docImg/detection.mov")

    clip1 = videoClip.subclip(1,25).resize(0.5)
    # clip2 = videoClip.subclip(52,62).resize(0.3)
    # finalclip = concatenate_videoclips([clip1, clip2])
    clip1.write_gif("/Users/ZhouTang/Desktop/docImg/detection_8.gif", fps=10)


