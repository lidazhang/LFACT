
import pickle
from tensorflow import Session
import os
import tensorflow as tf



def main(save_path, sess):

    if not os.path.exists(save_path):
        with open(save_path, "wb") as file1:

            variables = tf.trainable_variables()
            values = sess.run(variables)
            pickle.dump({var.name: val for var, val in zip(variables, values)}, file1)
    else:
        v_dic = {v.name: v for v in tf.trainable_variables()}

        for key, value in pickle.load(open(save_path, "rb")).items():
            #print key, ':', value
            print(key)
            sess.run(tf.assign(v_dic[key], value))


def load_np(save_path):

    if not os.path.exists(save_path):
        raise Exception("No saved weights at that location")
    else:
        v_dict = pickle.load(open(save_path, "rb"))
        for key in v_dict.keys():
            print("Key name: " + key)

    return v_dict


if __name__ == '__main__':
    a=load_np('./round1')
    #from sys import argv
    #exit(main(argv))