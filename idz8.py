import numpy as np
import matplotlib.pyplot as plt

def student():
    print("\033[35m")
    print("Студент: Каримов")
    print("Номер задачи: 8")
    print("\033[0m")

student()

np.set_printoptions(suppress=True, precision=6)

# ====================== ЧАСТЬ 1. Кубический сплайн ======================
# вариант 10: X* = -0.5
xs = -0.5
X1 = np.array([-3.0, -1.0, 1.0, 3.0, 5.0])
Y1 = np.array([-1.2490, -0.78540, 0.78540, 1.2490, 1.3734])
n = len(X1) - 1          
h = np.diff(X1)          


def progonka(a, b, c, d):

    m = len(d)
    P = np.zeros(m)
    Q = np.zeros(m)
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    for i in range(1, m):
        z = b[i] + a[i] * P[i - 1]
        P[i] = -c[i] / z if i < m - 1 else 0
        Q[i] = (d[i] - a[i] * Q[i - 1]) / z
    y = np.zeros(m)
    y[-1] = Q[-1]
    for i in range(m - 2, -1, -1):
        y[i] = P[i] * y[i + 1] + Q[i]
    return y


def koeff_splajna():

    m = n - 1
    a_d = np.zeros(m)
    b_d = np.zeros(m)
    c_d = np.zeros(m)
    r = np.zeros(m)
    for k in range(m):
        i = k + 1
        a_d[k] = h[i - 1]
        b_d[k] = 2 * (h[i - 1] + h[i])
        c_d[k] = h[i]
        r[k] = 3 * ((Y1[i + 1] - Y1[i]) / h[i] - (Y1[i] - Y1[i - 1]) / h[i - 1])

    print("Система для c_1, c_2, c_3 (трехдиагональная матрица):")
    print(np.diag(b_d) + np.diag(a_d[1:], -1) + np.diag(c_d[:-1], 1))
    print("Правая часть:", r)

    c = np.zeros(n + 1)
    c[1:n] = progonka(a_d, b_d, c_d, r)

    a = Y1[:n].copy()
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b[i] = (Y1[i + 1] - Y1[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    return a, b, c[:n], d, c


def znachenie(x, a, b, c, d):

    i = np.searchsorted(X1, x) - 1
    i = min(max(i, 0), n - 1)
    dx = x - X1[i]
    return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3, i


print("=" * 60)
print("ЧАСТЬ 1. Кубический сплайн")
print("=" * 60)

a, b, c, d, c_all = koeff_splajna()

print("\nКоэффициенты сплайна:")
for i in range(n):
    print(f"[{X1[i]:5.1f}; {X1[i+1]:4.1f}]: a = {a[i]:.6f}, b = {b[i]:.6f}, c = {c[i]:.6f}, d = {d[i]:.6f}")

A = np.diag(2 * (h[:-1] + h[1:])) + np.diag(h[1:-1], -1) + np.diag(h[1:-1], 1)
rr = 3 * (np.diff(Y1)[1:] / h[1:] - np.diff(Y1)[:-1] / h[:-1])
print("\nПроверка np.linalg.solve, c_1..c_3 =", np.linalg.solve(A, rr))
print("Прогонка,                  c_1..c_3 =", c_all[1:n])


print("\nПроверка непрерывности во внутренних узлах:")
for i in range(1, n):
    dx = h[i - 1]
    left = a[i - 1] + b[i - 1] * dx + c[i - 1] * dx ** 2 + d[i - 1] * dx ** 3
    print(f"x = {X1[i]:4.1f}: слева = {left:.6f}, справа = {a[i]:.6f}")

S, i = znachenie(xs, a, b, c, d)
print(f"\nX* = {xs} лежит на отрезке [{X1[i]}; {X1[i+1]}]")
print(f"S(X*) = {S:.6f}")
print(f"Для сравнения arctg(X*) = {np.arctan(xs):.6f}")


# ====================== ЧАСТЬ 2. МНК ======================
X2 = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
Y2 = np.array([-1.3734, -1.249, -0.7854, 0.7854, 1.249, 1.3734])


def normalnaya_sistema(m):
    # sum_k a_k * sum_i x_i^(k+j) = sum_i y_i x_i^j
    A = np.zeros((m + 1, m + 1))
    bb = np.zeros(m + 1)
    for j in range(m + 1):
        for k in range(m + 1):
            A[j, k] = np.sum(X2 ** (j + k))
        bb[j] = np.sum(Y2 * X2 ** j)
    return A, bb


def mnk(m):
    A, bb = normalnaya_sistema(m)
    print(f"\nМногочлен степени {m}")
    print("Матрица нормальной системы:")
    print(A)
    print("Правая часть:", bb)
    coef = np.linalg.solve(A, bb)          # a_0, a_1, ..., a_m
    print("Коэффициенты a_0..a_m:", coef)
    print("Проверка np.polyfit:  ", np.polyfit(X2, Y2, m)[::-1])
    return coef


def mnogochlen(coef, x):
    return sum(coef[k] * x ** k for k in range(len(coef)))


def summa_kvadratov(coef):
    # Ф = sum (P(x_i) - y_i)^2
    return np.sum((mnogochlen(coef, X2) - Y2) ** 2)


def vyvod_mnogochlena(coef):
    s = "P(x) = " + " + ".join(f"({cc:.6f})x^{k}" for k, cc in enumerate(coef))
    print(s)


def grafik(c1, c2):
    t = np.linspace(-5.5, 5.5, 400)
    plt.plot(X2, Y2, "ko", label="таблица")
    plt.plot(t, mnogochlen(c1, t), label="многочлен 1-й степени")
    plt.plot(t, mnogochlen(c2, t), label="многочлен 2-й степени")
    plt.axhline(0, color="k", lw=0.5)
    plt.axvline(0, color="k", lw=0.5)
    plt.grid()
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Аппроксимация методом наименьших квадратов")
    plt.show()


print("\n" + "=" * 60)
print("ЧАСТЬ 2. Метод наименьших квадратов")
print("=" * 60)

k1 = mnk(1)
vyvod_mnogochlena(k1)
Phi1 = summa_kvadratov(k1)
print(f"Ф1 = {Phi1:.6f}")

k2 = mnk(2)
vyvod_mnogochlena(k2)
Phi2 = summa_kvadratov(k2)
print(f"Ф2 = {Phi2:.6f}")

print("\nИтог:")
print(f"Сплайн: S(X*) = {S:.6f}")
print(f"МНК степень 1: Ф = {Phi1:.6f}")
print(f"МНК степень 2: Ф = {Phi2:.6f}")

grafik(k1, k2)