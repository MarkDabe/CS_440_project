import itertools
import numpy as np
import os


def get_num_lines(file):
    return sum(1 for line in open(file))


def construct_data(image_file, label_file):

    if not os.path.isfile(image_file) or not os.path.isfile(label_file):
        print("Input files do not exist")
        return None

    num_lines = get_num_lines(label_file)
    num_lines_images = get_num_lines(image_file)

    if num_lines != int(num_lines_images/28):
        print("Number of lines mismatch")
        return None

    image_matrix = np.zeros(shape=(num_lines,28,28))
    label_matrix = np.zeros(shape=num_lines)

    with open(image_file, "r") as open_image_file:
        line_handle = 0
        matrix_index = 0
        while line_handle < (num_lines_images - 27):
            hz_index = 0
            for line in itertools.islice(open_image_file, 0,  28):
                vt_index = 0
                for char in line:
                    if char == '#' or char == "+":
                            image_matrix[matrix_index][hz_index][vt_index] = 1
                    vt_index += 1
                hz_index += 1
            matrix_index += 1
            line_handle += 28

    with open(label_file, "r") as open_label_file:
        matrix_index = 0
        for line in open_label_file:
            label_matrix[matrix_index] = line.replace("\n", "")
            matrix_index += 1

    return image_matrix.astype(int), label_matrix.astype(int)


def digit_training_data(image_file="data/digitdata/trainingimages",
                            label_file="data/digitdata/traininglabels"):
    return construct_data(image_file, label_file)


def digit_test_data(image_file="data/digitdata/testimages",
                            label_file="data/digitdata/testlabels"):
    return construct_data(image_file, label_file)


def digit_validation_data(image_file="data/digitdata/validationimages",
                            label_file="data/digitdata/validationlabels"):
    return construct_data(image_file, label_file)


def face_training_data(image_file="data/digitdata/trainingimages",
                            label_file="data/digitdata/traininglabels"):
    return construct_data(image_file, label_file)


def face_test_data(image_file="data/digitdata/testimages",
                            label_file="data/digitdata/testlabels"):
    return construct_data(image_file, label_file)


def face_validation_data(image_file="data/digitdata/validationimages",
                            label_file="data/digitdata/validationlabels"):
    return construct_data(image_file, label_file)
