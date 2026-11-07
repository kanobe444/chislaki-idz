import numpy as np

def student():
    print("\033[35m")
    print("Студент: Каримов")
    print("Номер задачи: 8 (часть 1)")
    print("\033[0m")

student()

np.set_printoptions(suppress=True, precision=6)


xs = -0.5
X = np.array([-3.0, -1.0, 1.0, 3.0, 5.0])
F = np.array([-1.2490, -0.78540, 0.78540, 1.2490, 1.3734])
n = len(X) - 1         
h = np.diff(X)          # h_i = x_(i+1) - x_i


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
        r[k] = 3 * ((F[i + 1] - F[i]) / h[i] - (F[i] - F[i - 1]) / h[i - 1])

    print("Система для c_1, c_2, c_3 (трехдиагональная матрица):")
    print(np.diag(b_d) + np.diag(a_d[1:], -1) + np.diag(c_d[:-1], 1))
    print("Правая часть:", r)

    c = np.zeros(n + 1)
    c[1:n] = progonka(a_d, b_d, c_d, r)

    a = F[:n].copy()
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b[i] = (F[i + 1] - F[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    return a, b, c[:n], d, c


def znachenie(x, a, b, c, d):

    i = np.searchsorted(X, x) - 1
    i = min(max(i, 0), n - 1)
    dx = x - X[i]
    return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3, i


a, b, c, d, c_all = koeff_splajna()

print("\nКоэффициенты сплайна:")
for i in range(n):
    print(f"[{X[i]:5.1f}; {X[i+1]:4.1f}]: a = {a[i]:.6f}, b = {b[i]:.6f}, c = {c[i]:.6f}, d = {d[i]:.6f}")

A = np.diag(2 * (h[:-1] + h[1:])) + np.diag(h[1:-1], -1) + np.diag(h[1:-1], 1)
rr = 3 * (np.diff(F)[1:] / h[1:] - np.diff(F)[:-1] / h[:-1])
print("\nПроверка np.linalg.solve, c_1..c_3 =", np.linalg.solve(A, rr))
print("Прогонка,                  c_1..c_3 =", c_all[1:n])


print("\nПроверка непрерывности в внутренних узлах:")
for i in range(1, n):
    dx = h[i - 1]
    left = a[i - 1] + b[i - 1] * dx + c[i - 1] * dx ** 2 + d[i - 1] * dx ** 3
    print(f"x = {X[i]:4.1f}: слева = {left:.6f}, справа = {a[i]:.6f}")

S, i = znachenie(xs, a, b, c, d)
print(f"\nX* = {xs} лежит на отрезке [{X[i]}; {X[i+1]}]")
print(f"S(X*) = {S:.6f}")
print(f"Для сравнения arctg(X*) = {np.arctan(xs):.6f}")