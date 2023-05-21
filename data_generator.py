import time

import numpy as np

from rovio_lib import Rovio
from rovio_lib.rovio_connection import ROVIO_IP, ROVIO_LOGIN, ROVIO_PASS


class DataGenerator(object):
    '''
    class that is used to generate data
    '''

    def __init__(self) -> None:
        self.rovio = Rovio(ROVIO_IP, username=ROVIO_LOGIN,
                           password=ROVIO_PASS, port=80)
        self.random_state = np.random.RandomState(42)

    def get_data(self):
        # convert RGB to monochrome
        im1 = np.dot(self.rovio.camera.get_frame(), [
                     0.2989, 0.5870, 0.1140]).astype(np.uint8)
        vector = self.get_random_vector()
        self.perform_step(vector)
        im2 = np.dot(self.rovio.camera.get_frame(), [
                     0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return im1, im2, vector

    def get_random_vector(self):
        '''
        Returns a random vector:
        command number (0-9),
        speed (1-10),
        angle in rovio units or a time in seconds (0-4)
        Note: 1 rovio unit is ~11 degrees
        '''
        return (self.random_state.randint(0, 10),
                self.random_state.randint(1, 11),
                self.random_state.rand() * 4)

    def perform_step(self, vector):
        if vector[0] in [4, 5]:
            self.rovio.api.manual_drive(
                vector[0] + 1, speed=vector[1], angle=vector[2])
        else:
            self.rovio.api.manual_drive(vector[0] + 1, speed=vector[1])
            time.sleep(vector[2])
            self.rovio.api.manual_drive(0)
