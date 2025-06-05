import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Paramètres
a = 1.0       # demi-largeur du puits
V0 = 50.0     # profondeur du puits
L = 5.0       # taille du domaine spatial
N = 1000      # nombre de points pour la discrétisation

x = np.linspace(0, L, N)
dx = x[1] - x[0]

# Potentiel : puits centré à L/2, de profondeur -V0
V = np.zeros_like(x)
V[np.abs(x - L/2) <= a] = -V0

# Hamiltonien
kin = -2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)
kin /= dx**2
H = -kin + np.diag(V)  # H = T + V

# Diagonalisation
E, psi = eigh(H)

# Tracé des états stationnaires
plt.figure(figsize=(12, 8))

# On choisit d'afficher les 10 premiers états
for n in range(10):
    psi_n = psi[:, n]
    psi_n /= np.sqrt(np.sum(psi_n**2) * dx)  # normalisation
    plt.plot(x, psi_n + E[n], label=f'n={n}, E={E[n]:.2f}')

plt.plot(x, V, 'k--', label='V(x)')  # potentiel
plt.title('États stationnaires du puits fini (10 premiers états)')
plt.xlabel('x')
plt.ylabel('Énergie')
plt.legend()
plt.grid()
plt.show()
