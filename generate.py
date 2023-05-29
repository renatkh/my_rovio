from data_generator import DataGenerator
import numpy as np
import os
import uuid


for i in range(3):
    generator = DataGenerator()
    store_path = '/Volumes/Documents/datasets/rovio/'
    path_exists = os.path.exists(store_path)
    assert path_exists, 'Path does not exist'
    num_steps = 100
    im1, im2, vector = generator.get_data()
    im1 = np.expand_dims(im1, axis=2)
    im2 = np.expand_dims(im2, axis=2)
    vector = np.expand_dims(vector, axis=0)
    for i in range(num_steps - 1):
        _im1, _im2, _vector = generator.get_data()
        im1 = np.concatenate((im1, np.expand_dims(_im1, axis=2)), axis=2)
        im2 = np.concatenate((im2, np.expand_dims(_im2, axis=2)), axis=2)
        vector = np.concatenate((vector, np.expand_dims(_vector, axis=0)), axis=0)
        print(f'{i+1}/{num_steps}')
    np.savez_compressed(os.path.join(store_path, str(
        uuid.uuid4())), im1=im1, im2=im2, vector=vector)
