import numpy as np
def student():
    print("\033[33m")
    print("Студент: Каримов")
    print("Номер задачи: 3")
    print("\033[0m")

student()



np.set_printoptions(suppress=True)



A = np.array([
    [28,  9,  -3, -7],
    [-5, 21,  -5, -3],
    [-8,  1, -16,  5],
    [ 0, -2,   5,  8],
], dtype=float)
b = np.array([-159, 63, -45, 24], dtype=float)
n = len(b)
eps = 0.01

#x = B x + c
B = -A / np.diag(A).reshape(-1, 1)
np.fill_diagonal(B, 0)
c = b / np.diag(A)


def prostie_iterats(B, c, eps):
    x = c.copy()                      
    k = 0
    while True:
        x_new = B @ x + c
        k += 1
        if np.max(np.abs(x_new - x)) < eps:
            return x_new, k
        x = x_new


def zeidel(B, c, eps):
    x = c.copy()
    k = 0
    while True:
        x_new = x.copy()
        for i in range(n):


            x_new[i] = B[i] @ x_new + c[i]
        k += 1
        if np.max(np.abs(x_new - x)) < eps:
            return x_new, k
        x = x_new


x1, k1 = prostie_iterats(B, c, eps)
x2, k2 = zeidel(B, c, eps)

print("Метод простых итераций: x =", np.round(x1, 4), ", вышло итераций:", k1)
print("Метод Зейделя:          x =", np.round(x2, 4), ", вышлоо итераций:", k2)