import random


def initMatrix(n):
    randomInts = list(range(1, n * n + 1))  # generate random number
    random.shuffle(randomInts)  # shuffle random numbers

    # task: create a n by n matrix and fill it with the random numbers
    mat = []
    for i in range(n):
        # 1 line code
        mat.append([])
        for j in range(n):
            # 1 line code
            mat[i].append(randomInts[i * n + j])

    return mat


def updateMatrix(mat, bn):
    for idx, i in enumerate(mat):
        print(idx, i)
        for jdx, j in enumerate(mat):
            if mat[idx][jdx] == bn:
                mat[idx][jdx] = "*"


def printMatrix(mat):
    for l in mat:
        print('\t'.join(map(str, l)))
