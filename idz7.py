import numpy as np

def student():
    print("\033[35m")
    print("Студент: Каримов")
    print("Номер задачи: 7")
    print("\033[0m")

student()

np.set_printoptions(suppress=True, precision=6)


xs = -0.5

X_a = np.array([-3, -1, 1, 3], dtype=float)
X_b = np.array([-3, 0, 1, 3], dtype=float)


def f(x):
    return np.arctan(x)


def d4f(x):
    # 4 производная arctg x
    return 24 * x * (1 - x ** 2) / (1 + x ** 2) ** 4


def lagranzh(X, Y, x):
    # P(x) = sum Y_i * l_i(x)
    n = len(X)
    P = 0
    for i in range(n):
        l = 1
        for j in range(n):
            if j != i:
                l *= (x - X[j]) / (X[i] - X[j])
        P += Y[i] * l
    return P


def razd_razn(X, Y):

    n = len(X)
    T = np.zeros((n, n))
    T[:, 0] = Y
    for j in range(1, n):
        for i in range(n - j):
            T[i, j] = (T[i + 1, j - 1] - T[i, j - 1]) / (X[i + j] - X[i])
    return T


def nyuton(X, T, x):
    # P(x) = f(x0) + f(x0,x1)(x-x0) + f(x0,x1,x2)(x-x0)(x-x1) + ...
    n = len(X)
    P = 0
    proizv = 1
    for k in range(n):
        P += T[0, k] * proizv
        proizv *= (x - X[k])
    return P


def omega(X, x):
    return np.prod(x - X)


def ocenka(X, x):

    t = np.linspace(X[0], X[-1], 100001)
    M4 = np.max(np.abs(d4f(t)))
    return M4 / 24 * abs(omega(X, x)), M4


def reshenie(nazv, X):
    Y = f(X)
    print("=" * 60)
    print(f"Вариант {nazv}: X =", X)
    print("Y =", Y)

    # Лагранж
    PL = lagranzh(X, Y, xs)

    # Ньютон
    T = razd_razn(X, Y)
    print("\nТаблица разделенных разностей:")
    print(T)
    PN = nyuton(X, T, xs)

    tochn = f(xs)
    pogr_L = abs(tochn - PL)
    pogr_N = abs(tochn - PN)

    print(f"\nТочное значение f(X*) = {tochn:.6f}")
    print(f"Лагранж: P(X*) = {PL:.6f}, погрешность = {pogr_L:.6f}")
    print(f"Ньютон:  P(X*) = {PN:.6f}, погрешность = {pogr_N:.6f}")

    ots, M4 = ocenka(X, xs)
    print(f"\n|omega(X*)| = {abs(omega(X, xs)):.6f}")
    print(f"M4 = max|f''''| на [{X[0]}; {X[-1]}] = {M4:.6f}")
    print(f"Оценка (3.9) = {ots:.6f}")
    print(f"Фактическая погрешность <= оценки: {pogr_L <= ots}")
    return pogr_L, pogr_N


reshenie("а", X_a)
reshenie("б", X_b)