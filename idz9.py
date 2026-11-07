import numpy as np

def student():
    print("\033[35m")
    print("Студент: Каримов")
    print("Номер задачи: 9")
    print("\033[0m")

student()

np.set_printoptions(suppress=True, precision=6)

# ====================== ЧАСТЬ 1. Производные ======================
# вариант 10: X* = 1.0
xs = 1.0
X1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
Y1 = np.array([0.0, 0.97943, 1.8415, 2.4975, 2.9093])
h0 = X1[1] - X1[0]                       
i0 = int(np.argmin(np.abs(X1 - xs)))     # номер узла(x = X*)


def razn_vpered(i):
    # y' ≈ (y_(i+1) - y_i) / h, порядок O(h)
    return (Y1[i + 1] - Y1[i]) / h0


def razn_nazad(i):
    # y' ≈ (y_i - y_(i-1)) / h, порядок O(h)
    return (Y1[i] - Y1[i - 1]) / h0


def centr_pervaya(i):
    # y' ≈ (y_(i+1) - y_(i-1)) / (2h), порядок O(h2)
    return (Y1[i + 1] - Y1[i - 1]) / (2 * h0)


def centr_vtoraya(i):
    # y'' ≈ (y_(i+1) - 2 y_i + y_(i-1)) / h^2, порядок O(h2)
    return (Y1[i + 1] - 2 * Y1[i] + Y1[i - 1]) / h0 ** 2


def pyat_tochek_pervaya(i):
    # y' ≈ (y_(i-2) - 8 y_(i-1) + 8 y_(i+1) - y_(i+2)) / (12 h), порядок O(h4)
    return (Y1[i - 2] - 8 * Y1[i - 1] + 8 * Y1[i + 1] - Y1[i + 2]) / (12 * h0)


def pyat_tochek_vtoraya(i):
    # y'' ≈ (-y_(i-2) + 16 y_(i-1) - 30 y_i + 16 y_(i+1) - y_(i+2)) / (12 h^2), порядок O(h4)
    return (-Y1[i - 2] + 16 * Y1[i - 1] - 30 * Y1[i] + 16 * Y1[i + 1] - Y1[i + 2]) / (12 * h0 ** 2)


print("=" * 60)
print("ЧАСТЬ 1. Производные таблично заданной функции")
print("=" * 60)

print(f"Шаг h = {h0}, X* = {xs} совпадает с узлом x_{i0} = {X1[i0]}")
print("Таблица:")
print("x =", X1)
print("y =", Y1)

# первая производная
d1_v = razn_vpered(i0)
d1_n = razn_nazad(i0)
d1_c = centr_pervaya(i0)
d1_5 = pyat_tochek_pervaya(i0)

print("\nПервая производная y'(X*):")
print(f"Разность вперед:         {d1_v:.6f}")
print(f"Разность назад:          {d1_n:.6f}")
print(f"Центральная (O(h^2)):    {d1_c:.6f}")
print(f"По 5 точкам (O(h^4)):    {d1_5:.6f}")

# вторая производная
d2_c = centr_vtoraya(i0)
d2_5 = pyat_tochek_vtoraya(i0)

print("\nВторая производная y''(X*):")
print(f"Центральная (O(h^2)):    {d2_c:.6f}")
print(f"По 5 точкам (O(h^4)):    {d2_5:.6f}")


t1 = 1 + np.cos(xs)
t2 = -np.sin(xs)
print("\nПроверка для y = x + sin x:")
print(f"y'(X*)  точное = {t1:.6f}, погрешность (5 точек) = {abs(d1_5 - t1):.6f}")
print(f"y''(X*) точное = {t2:.6f}, погрешность (5 точек) = {abs(d2_5 - t2):.6f}")

print("\nОтвет (часть 1):")
print(f"y'(X*)  ≈ {d1_5:.4f}")
print(f"y''(X*) ≈ {d2_5:.4f}")


# ====================== ЧАСТЬ 2. Интегралы ======================
# вариант 10: y = x^2 / (x^2 + 16), X0 = 0, X1 = 2, h1 = 0.5, h2 = 0.25
a, b = 0.0, 2.0
h1, h2 = 0.5, 0.25


def f(x):
    return x ** 2 / (x ** 2 + 16)


def d2f(x):
    # y'' = 32 (16 - 3x^2) / (x^2 + 16)^3
    return 32 * (16 - 3 * x ** 2) / (x ** 2 + 16) ** 3


def d4f(x):
    # y'''' = -384 (5x^4 - 160x^2 + 256) / (x^2 + 16)^5
    return -384 * (5 * x ** 4 - 160 * x ** 2 + 256) / (x ** 2 + 16) ** 5


def pryamougolniki(h):
    # средние прямоугольники
    n = int(round((b - a) / h))
    x = a + h * (np.arange(n) + 0.5)
    return h * np.sum(f(x))


def trapecii(h):
    n = int(round((b - a) / h))
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (y[0] / 2 + np.sum(y[1:-1]) + y[-1] / 2)


def simpson(h):
    n = int(round((b - a) / h))    
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])


def runge_romberg(F1, F2, p):

    return F2 + (F2 - F1) / ((h1 / h2) ** p - 1)


def maks(g):

    t = np.linspace(a, b, 100001)
    return np.max(np.abs(g(t)))


def ocenka_pryam(h, M2):
    # (3.28) R = (b - a) h^2 M2 / 24
    return (b - a) * h ** 2 * M2 / 24


def ocenka_trap(h, M2):
    # (3.30) R = (b - a) h^2 M2 / 12
    return (b - a) * h ** 2 * M2 / 12


def ocenka_simpson(h, M4):
    # (3.33) R = (b - a) h^4 M4 / 180
    return (b - a) * h ** 4 * M4 / 180


print("\n" + "=" * 60)
print("ЧАСТЬ 2. Определенный интеграл")
print("=" * 60)

M2 = maks(d2f)
M4 = maks(d4f)
tochno = 2 - 4 * np.arctan(0.5)   

print(f"Интеграл от y = x^2/(x^2+16) на [{a}; {b}]")
print(f"h1 = {h1}, h2 = {h2}")
print(f"M2 = max|y''|   = {M2:.6f}")
print(f"M4 = max|y''''| = {M4:.6f}")

# значения методов
P1, P2 = pryamougolniki(h1), pryamougolniki(h2)
T1, T2 = trapecii(h1), trapecii(h2)
S1, S2 = simpson(h1), simpson(h2)

print("\nЗначения интеграла:")
print(f"{'метод':<22}{'h1':>12}{'h2':>12}")
print(f"{'Прямоугольники':<22}{P1:>12.6f}{P2:>12.6f}")
print(f"{'Трапеции':<22}{T1:>12.6f}{T2:>12.6f}")
print(f"{'Симпсон':<22}{S1:>12.6f}{S2:>12.6f}")


Pr = runge_romberg(P1, P2, 2)
Tr = runge_romberg(T1, T2, 2)
Sr = runge_romberg(S1, S2, 4)

print("\nУточнение по Рунге - Ромбергу:")
print(f"Прямоугольники: {Pr:.6f}")
print(f"Трапеции:       {Tr:.6f}")
print(f"Симпсон:        {Sr:.6f}")


print("\nОценка погрешности по формулам (3.28), (3.30), (3.33):")
print(f"{'метод':<22}{'h1':>12}{'h2':>12}")
print(f"{'Прямоугольники':<22}{ocenka_pryam(h1, M2):>12.6f}{ocenka_pryam(h2, M2):>12.6f}")
print(f"{'Трапеции':<22}{ocenka_trap(h1, M2):>12.6f}{ocenka_trap(h2, M2):>12.6f}")
print(f"{'Симпсон':<22}{ocenka_simpson(h1, M4):>12.8f}{ocenka_simpson(h2, M4):>12.8f}")


print(f"\nТочное значение F = 2 - 4 arctg(0.5) = {tochno:.6f}")
print("Фактическая погрешность:")
print(f"{'метод':<22}{'h1':>12}{'h2':>12}{'Рунге-Ромберг':>16}")
print(f"{'Прямоугольники':<22}{abs(P1 - tochno):>12.6f}{abs(P2 - tochno):>12.6f}{abs(Pr - tochno):>16.8f}")
print(f"{'Трапеции':<22}{abs(T1 - tochno):>12.6f}{abs(T2 - tochno):>12.6f}{abs(Tr - tochno):>16.8f}")
print(f"{'Симпсон':<22}{abs(S1 - tochno):>12.8f}{abs(S2 - tochno):>12.8f}{abs(Sr - tochno):>16.8f}")

print("\nОтвет (часть 2):")
print(f"F ≈ {Sr:.6f} (Симпсон, уточнено по Рунге - Ромбергу)")