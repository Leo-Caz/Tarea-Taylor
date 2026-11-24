
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CARPETA = Path(__file__).resolve().parent


def taylor_sin(x, n):
    """Polinomio de Maclaurin para sin(x), de grado impar n."""
    if n < 1 or n % 2 == 0:
        raise ValueError('n debe ser impar y mayor o igual que 1')
    x = np.asarray(x, dtype=float)
    termino = x.copy()
    suma = termino.copy()
    for k in range((n - 1) // 2):
        termino = termino * (-x**2) / ((2*k + 2)*(2*k + 3))
        suma = suma + termino
    return float(suma) if suma.ndim == 0 else suma


def serie_geometrica(x, N):
    """S_N(x) = sum_{k=0}^N (-1)^k*x^(2*k), con recurrencia."""
    if N < 0:
        raise ValueError('N debe ser no negativo')
    x = np.asarray(x, dtype=float)
    termino = np.ones_like(x, dtype=float)
    suma = termino.copy()
    for _ in range(N):
        termino = termino * (-x**2)
        suma = suma + termino
    return float(suma) if suma.ndim == 0 else suma


def taylor_exp(x, n):
    """Suma 1 + x + x^2/2! + ... + x^n/n!, actualizando el término."""
    suma = 1.0
    termino = 1.0
    for k in range(1, n + 1):
        termino *= x / k
        suma += termino
    return suma


def grafica_1():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 2001)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, np.sin(x), label='sin(x)', color='black', linewidth=2.5)
    for n in [1, 3, 5, 7, 9]:
        ax.plot(x, taylor_sin(x, n), label=f'P{n}(x)')
    ax.set(xlabel='x (radianes)', ylabel='Valor',
           title='Gráfica 1. sin(x) y polinomios de Taylor', ylim=(-2, 2))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CARPETA / 'grafica_1_seno.png', dpi=180)
    plt.close(fig)


def grafica_2():
    grados = list(range(1, 22, 2))
    fig, ax = plt.subplots(figsize=(10, 6))
    for a, b, etiqueta in [(-np.pi, np.pi, '[-π, π]'),
                          (-np.pi / 4, np.pi / 4, '[-π/4, π/4]')]:
        x = np.linspace(a, b, 2001)
        errores = [float(np.max(np.abs(np.sin(x) - taylor_sin(x, n))))
                   for n in grados]
        ax.semilogy(grados, errores, marker='o', label=etiqueta)
        menores = [n for n, e in zip(grados, errores) if e < 1e-6]
        print('Primer grado con error < 1e-6 en', etiqueta, ':',
              menores[0] if menores else 'no encontrado')
    ax.set(xlabel='Grado impar n', ylabel='Error máximo (escala logarítmica)',
           title='Gráfica 2. Error máximo de los polinomios de Taylor')
    ax.set_xticks(grados)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CARPETA / 'grafica_2_errores.png', dpi=180)
    plt.close(fig)


def grafica_3():
    x = np.linspace(-1.5, 1.5, 2001)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, 1 / (1 + x**2), label='f(x) = 1/(1+x²)',
            color='black', linewidth=2.5)
    for N in [2, 5, 10, 20]:
        ax.plot(x, serie_geometrica(x, N), label=f'S{N}(x)')
    ax.set(xlabel='x', ylabel='Valor',
           title='Gráfica 3. Serie geométrica y sumas parciales',
           ylim=(-0.5, 1.5))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CARPETA / 'grafica_3_geometrica.png', dpi=180)
    plt.close(fig)


def main():
    print('SECCIÓN 2')
    print('P9(pi) =', f'{taylor_sin(math.pi, 9):.6f}')
    grafica_1()
    grafica_2()
    print('\nSECCIÓN 3')
    print('S40(0.9) =', f'{serie_geometrica(0.9, 40):.12f}')
    print('S40(1.1) =', f'{serie_geometrica(1.1, 40):.12f}')
    grafica_3()
    print('\nSECCIÓN 4')
    negativo = taylor_exp(-20, 100)
    positivo = taylor_exp(20, 100)
    print('P100(-20) =', f'{negativo:.15g}')
    print('math.exp(-20) =', f'{math.exp(-20):.15g}')
    print('1/P100(20) =', f'{1 / positivo:.15g}')
    termino = 1.0
    mayor = 1.0
    k_mayor = 0
    for k in range(1, 101):
        termino *= -20 / k
        if abs(termino) > mayor:
            mayor = abs(termino)
            k_mayor = k
    print('Mayor valor absoluto de término =',
          f'{mayor:.12f}', 'en k =', k_mayor)
    print('\nGráficas guardadas en:', CARPETA)


if __name__ == '__main__':
    main()
