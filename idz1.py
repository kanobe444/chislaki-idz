import numpy as np
def student():
    print("\033[32m")
    print("Студент: Каримов")
    print("Номер задачи: 1")
    print("\033[0m")

student()

a = [0,  6, -3, -9, -5] 
b = [-7, 12, 5, 21, -6]  
c = [-6, 0, 0,  8,  0]  
d = [-75, 126, 13, -40, -24]
n = len(d)

def reshenie(a, b, c, d):
    n = len(d)
    alpha = [0]*n; beta = [0]*n


    alpha[0] = -c[0]/b[0]
    beta[0] = d[0]/b[0]
    for i in range(1, n):
        den = b[i] + a[i]*alpha[i-1]
        alpha[i] = -c[i]/den
        beta[i] = (d[i] - a[i]*beta[i-1])/den


    x = [0]*n
    x[-1] = beta[-1]
    for i in range(n-2, -1, -1):
        x[i] = alpha[i]*x[i+1] + beta[i]
    return x, alpha, beta

x, alpha, beta = reshenie(a, b, c, d)
print("alpha =", np.round(alpha, 6))
print("beta  =", np.round(beta, 6))
print("x     =", np.round(x, 6))

