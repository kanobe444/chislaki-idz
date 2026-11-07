import numpy as np
import matplotlib.pyplot as plt

def student():
    print("\033[35m")
    print("Студент: Каримов")
    print("Номер задачи: 6")
    print("\033[0m")

student()

np.set_printoptions(suppress=True, precision=6)

a = 1        
eps = 1e-4   


def F(x):
    #x1 - cos x2 = a, x2 - sin x1 = a
    return np.array([
        x[0] - np.cos(x[1]) - a,
        x[1] - np.sin(x[0]) - a,
    ])


def yakobi(x):
    return np.array([
        [1, np.sin(x[1])],
        [-np.cos(x[0]), 1],
    ])


def grafik():
    # x1 = a + cos x2  и  x2 = a + sin x1
    t = np.linspace(-1, 3, 400)
    plt.plot(a + np.cos(t), t, label="x1 = a + cos x2")
    plt.plot(t, a + np.sin(t), label="x2 = a + sin x1")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    plt.grid()
    plt.legend()
    plt.title("Графическое определение начального приближения")
    plt.show()


def metod_nyutona(x0, eps, max_iter=100):
    x = x0.copy()
    k = 0
    print("Метод Ньютона: x_(k+1) = x_k - W^(-1)(x_k) * F(x_k)")
    print("x0 =", x)
    while k < max_iter:
        dx = np.linalg.solve(yakobi(x), -F(x))  
        x_new = x + dx
        k += 1
        pogr = np.max(np.abs(x_new - x))
        print(f"\nИтерация {k}: x = {x_new}, max|x_k - x_(k-1)| = {pogr:.6f}")
        x = x_new
        if pogr < eps:
            break
    return x, k


grafik()

# по графику пересечение кривых приммерно (0.8; 1.7) оба значения +
x0 = np.array([0.8, 1.7])

x, k = metod_nyutona(x0, eps)

print("\nРезультат:")
print(f"x1 = {x[0]:.6f}, x2 = {x[1]:.6f}, итераций: {k}")
print("Невязка F(x) =", F(x))