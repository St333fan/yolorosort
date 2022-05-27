#!/usr/bin/env python3
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

# rospy for the subscriber

import rospy
# ROS Image message

from yolorosort.msg import fullBBox, singleBBox, cluster

import numpy as np
import numpy as np
from sklearn.cluster import DBSCAN


class subscriber:

    def __init__(self):

        self.sub = rospy.Subscriber('/usb_cam/image_raw/boundingboxes', fullBBox, self.callback)  # instantiate the Subscriber and Publisher
        self.pub = rospy.Publisher('/usb_cam/image_raw/clusterlabels', cluster, queue_size=10)

    def callback(self, data):
        print("working on cluster... ")

        i = 0
        for sBox in data.boxesWithAll:
            if i == 0:
                X = np.array([[sBox.x, sBox.y]])
                i = 1
            else:
                X = np.append(X, [[sBox.x, sBox.y]], axis=0)

        if i > 0:
            print(X)
            clustering = DBSCAN(eps=0.3, min_samples=2).fit(X)
            print(clustering.labels_)

        clu = cluster()
        clu.tampOfFullBBox = data.sstampOfOriginal
        clu.labels = clustering.labels_
        self.pub.publish(clu)


def main():
    obc = subscriber()
    rospy.init_node('cluster', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main()
