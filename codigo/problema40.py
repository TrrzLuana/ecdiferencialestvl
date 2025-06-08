import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parámetros
m = 5
g = 9.81
k = 0.05
v0 = 0
t0, t_end, h = 0, 15, 1
n_steps = int((t_end - t0) / h)

def dvdt(t, v):
    return -g + (k / m) * v**2

def heun_method(f, t0, y0, h, n):
    t_values = [t0]
    y_values = [y0]
    table = []

    for i in range(n):
        t, y = t_values[-1], y_values[-1]
        k1 = f(t, y)
        k2 = f(t + h, y + h * k1)
        y_next = y + h * (k1 + k2) / 2

        table.append([i, t, y, k1, k2, y_next])
        t_values.append(t + h)
        y_values.append(y_next)

    df = pd.DataFrame(table, columns=["i", "t", "y", "k1", "k2", "y_next"])
    return t_values, y_values, df

def rk4_method(f, t0, y0, h, n):
    t_values = [t0]
    y_values = [y0]
    table = []

    for i in range(n):
        t, y = t_values[-1], y_values[-1]
        k1 = f(t, y)
        k2 = f(t + h/2, y + h * k1 / 2)
        k3 = f(t + h/2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)
        y_next = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6

        table.append([i, t, y, k1, k2, k3, k4, y_next])
        t_values.append(t + h)
        y_values.append(y_next)

    df = pd.DataFrame(table, columns=["i", "t", "y", "k1", "k2", "k3", "k4", "y_next"])
    return t_values, y_values, df

# Aplicar métodos
t_heun, y_heun, tabla_heun = heun_method(dvdt, t0, v0, h, n_steps)
t_rk4, y_rk4, tabla_rk4 = rk4_method(dvdt, t0, v0, h, n_steps)

# Mostrar tablas
print("Tabla Heun:\n", tabla_heun)
print("\nTabla RK4:\n", tabla_rk4)

# Gráfico
plt.plot(t_heun, y_heun, label='Heun', marker='o')
plt.plot(t_rk4, y_rk4, label='RK4', marker='x')
plt.title("Ejercicio 40 – Caída libre con resistencia")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad v (m/s)")
plt.legend()
plt.grid(True)
plt.show()
