import numpy as np

RESULT_FILENAME = "result.txt"
RESULT_MATRIX_FILENAME = "result matrix.txt"
FIRST_MATRIX_FILENAME = "first matrix.txt"
SECOND_MATRIX_FILENAME = "second matrix.txt"


def check():
    first_matrix = np.loadtxt(FIRST_MATRIX_FILENAME, int)
    second_matrix = np.loadtxt(SECOND_MATRIX_FILENAME, int)
    loaded_result_matrix = np.loadtxt(RESULT_MATRIX_FILENAME, int)
    
    multiply = np.dot(first_matrix, second_matrix)

    if np.array_equal(multiply, loaded_result_matrix):
        print("Matrices are equal")
    else:
        print("Matrices are not equal")


if __name__ == "__main__":
    check()