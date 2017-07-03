#given a basis, visualize vectors

import numpy as np
import h5py
from mayavi import mlab
from mayavi.mlab import pipeline

class BasisVisualizer():
    def __init__(self, basis_file, obj_size):
        '''
        :param basis_file: location of basis file
        '''

        with h5py.File(basis_file) as f:
            self.W = f['A'][:]
            self.Mu = f['Mu'][:]

        self.obj_size = obj_size

        #need to initialize figure before TF can grab GPU

        self.fig = mlab.figure()

    def reconstruct(self, vec):
        '''
        :param vec: BEO embedding vector.
        :param cutoff:
        :return: reconstructed object.
        '''
        recon = np.dot(vec, self.W) - self.Mu
        return recon.reshape(self.obj_size)


    def visualize(self, object, cutoff=0.2):
        '''
        visualizes object.
        :param object: voxel grid.
        :param cutoff: threshold at which to display the voxel.
        :return:
        '''

        #preprocessing object
        object[object >= cutoff] = 1
        object[object < cutoff] = 0

        src = pipeline.scalar_field(object)
        mlab.gcf()
        pipeline.iso_surface(src)

        mlab.show()

