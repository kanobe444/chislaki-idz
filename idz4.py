import numpy as np
def student():
    print("\033[35m")
    print("Студент: Каримов")
    print("Номер задачи: 4")
    print("\033[0m")

student()



np.set_printoptions(suppress=True, precision=4)

A = np.array([
    [-7, -6,  8],
    [-6,  3, -7],
    [ 8, -7,  4],
], dtype=float)
n = len(A)
eps = 2  


def norm(A):
    
    return np.sqrt(np.sum(A ** 2) - np.sum(np.diag(A) ** 2))


def vrashenie(A, eps, max_iter=100):
    A = A.copy()
    V = np.eye(n)                       # сюда собств векторы
    k = 0
    print("Исходная t(A) =", round(norm(A), 4))
    while norm(A) >= eps and k < max_iter:
        # макс по модулю внедиагонал 
        B = np.abs(A - np.diag(np.diag(A)))
        i, j = np.unravel_index(np.argmax(B), B.shape)
        
        phi = 0.5 * np.arctan2(2 * A[i, j], A[i, i] - A[j, j])
        c, s = np.cos(phi), np.sin(phi)
        U = np.eye(n)
        U[i, i] = U[j, j] = c
        U[i, j] = -s
        U[j, i] = s
        A = U.T @ A @ U
        V = V @ U
        k += 1
        print(f"\nИтерация {k}: обнуляем a[{i+1}{j+1}], φ = {phi:.4f}, t(A) = {norm(A):.4f}")
        print(A)
    return np.diag(A), V, k


lam, V, k = vrashenie(A, eps)

print("\nСобственные значени:", lam)
print("Собственные векторы (столбцы  ):")
print(V)