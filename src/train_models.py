from models.lang_beo_rnn import BEORNN

#not using argparse for once


def run():
    data_dir = "../data"
    vector_dir = "{}/embedding_vectors".format(data_dir)
    annotations = "{}/annotated/debug/debug_original_100.txt".format(data_dir)
    model_loc = "models/saved/debug_100/model.ckpt"

    rnn = BEORNN(annotations, vector_dir, saver_loc=model_loc, epochs=500)
    rnn.train()

if __name__ == "__main__":
    run()