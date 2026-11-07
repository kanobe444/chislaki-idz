import numpy as np
import matplotlib.pyplot as plt

def student():
    print("\033[35m")
    print("Студент: Каримов")
    print("Номер задачи: 5")
    print("\033[0m")

student()

np.set_printoptions(suppress=True, precision=4)

eps = 1e-4   # точность взял 1e-4)


def f(x):
    return np.sin(x) - 2 * x ** 2 + 0.5


def df(x):
    return np.cos(x) - 4 * x


# для простой итерации
def fi(x):
    return np.sqrt((np.sin(x) + 0.5) / 2)


def dfi(x):
    return np.cos(x) / (4 * fi(x))


def grafik():

    x = np.linspace(0, 1.5, 300)
    plt.plot(x, np.sin(x) + 0.5, label="y = sin x + 0.5")
    plt.plot(x, 2 * x ** 2, label="y = 2x²")
    plt.axhline(0, color="k", lw=0.5)
    plt.grid()
    plt.legend()
    plt.title("Графическое определение корня")
    plt.show()


def otdelenie():

    print("Отделение корня (таблица значений f(x)):")
    for x in np.arange(0, 1.01, 0.1):
        print(f"x = {x:.1f}   f(x) = {f(x):.4f}")


def prostaya_iteraciya(x0, eps, max_iter=100):
    x = x0
    k = 0
    print("\nМетод простой итерации: x = sqrt((sin x + 0.5)/2)")
    print(f"Проверка сходимости: |φ'(x0)| = {abs(dfi(x0)):.4f} < 1")
    print(f"x0 = {x0}")
    while k < max_iter:
        x_new = fi(x)
        k += 1
        print(f"Итерация {k}: x = {x_new:.6f}, |x_k - x_(k-1)| = {abs(x_new - x):.6f}")
        if abs(x_new - x) < eps:
            x = x_new
            break
        x = x_new
    return x, k


def metod_nyutona(x0, eps, max_iter=100):
    x = x0
    k = 0
    print("\nМетод Ньютона: x_(k+1) = x_k - f(x_k)/f'(x_k)")
    print(f"x0 = {x0}")
    while k < max_iter:
        x_new = x - f(x) / df(x)
        k += 1
        print(f"Итерация {k}: x = {x_new:.6f}, |x_k - x_(k-1)| = {abs(x_new - x):.6f}")
        if abs(x_new - x) < eps:
            x = x_new
            break
        x = x_new
    return x, k


otdelenie()
grafik()


x0 = 0.8

x1, k1 = prostaya_iteraciya(x0, eps)
x2, k2 = metod_nyutona(x0, eps)

print("\nРезультаты:")
print(f"Простая итерация: x = {x1:.6f}, итераций: {k1}, f(x) = {f(x1):.6f}")
print(f"Ньютон:           x = {x2:.6f}, итераций: {k2}, f(x) = {f(x2):.6f}")