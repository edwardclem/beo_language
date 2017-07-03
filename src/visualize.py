from beo_projection.visualize_vectors import BasisVisualizer
from models.lang_beo_rnn import BEORNN
import scipy.io as sio


def run():
    data_dir = "../data"
    basis_loc = "{}/basis/Modelnet_Chairs_auto_basis_size_30.mat".format(data_dir)
    viz = BasisVisualizer(basis_loc, [30, 30, 30])

    # #loading vector for debug
    # mat = sio.loadmat("{}/embedding_vectors/chair_0890_1_vec.mat".format(data_dir))
    # vec = mat['projectedObVec']
    #
    # shape = viz.reconstruct(vec)
    #
    # viz.visualize(shape)

    #load RNN
    data_dir = "../data"
    vector_dir = "{}/embedding_vectors".format(data_dir)
    annotations = "{}/annotated/debug/debug_original_100.txt".format(data_dir)
    model_loc = "models/saved/debug_100/model.ckpt"

    rnn = BEORNN(annotations, vector_dir, load_loc=model_loc)
    predicted = rnn.phrase_to_vec("chair")

    shape = viz.reconstruct(predicted)

    viz.visualize(shape, cutoff=0.2)




if __name__ == "__main__":
    run()