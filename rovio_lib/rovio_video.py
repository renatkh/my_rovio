import cv2
import base64
import urllib.request as urllib2
import numpy as np


class ipCamera(object):

    def __init__(self, url, user=None, password=None):
        self.url = url
        auth_encoded = base64.encodebytes(
            f'{user}:{password}'.encode('ascii'))[:-1]

        self.req = urllib2.Request(self.url)
        self.req.add_header('Authorization', 'Basic %s' % auth_encoded)

    def get_frame(self):
        response = urllib2.urlopen(self.req)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, 1)
        return frame
