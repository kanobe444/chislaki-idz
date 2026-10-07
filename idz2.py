import numpy as np
def student():
    print("\033[36m")
    print("Студент: Каримов")
    print("Номер задачи: 2")
    print("\033[0m")

student()




np.set_printoptions(precision=6, suppress=True, linewidth=120)





A = np.array([
    [-7,  3, -4,  7],
    [ 8, -1, -7,  6],
    [ 9,  9,  3, -6],
    [-7, -9, -8, -5],
], dtype=float)
b = np.array([-126, 29, 27, 34], dtype=float)
n = len(b)


def metod_gauss(A, b):

    M = np.hstack([A.copy(), b.reshape(-1, 1)])  # тут расширенная матрицв
    det = 1.0




    for k in range(n):
        p = k + np.argmax(np.abs(M[k:, k]))      

        if p != k:
            M[[k, p]] = M[[p, k]]                
            det = -det                           
        det *= M[k, k]
        for i in range(k + 1, n):
            M[i, k:] -= M[i, k] / M[k, k] * M[k, k:]


    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, n] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i]
    return x, det


def gauss_obratka(A):

    M = np.hstack([A.copy(), np.eye(n)])
    for k in range(n):
        p = k + np.argmax(np.abs(M[k:, k]))

        if p != k:
            M[[k, p]] = M[[p, k]]
        M[k] /= M[k, k]                          
        for i in range(n):
            if i != k:
                M[i] -= M[i, k] * M[k]          
    return M[:, n:]


x, det = metod_gauss(A, b)
A_inv = gauss_obratka(A)

print("Решение x =", x)
print("опеределитель  =", round(det, 6))
print("обратная матрица =")
print(A_inv)
