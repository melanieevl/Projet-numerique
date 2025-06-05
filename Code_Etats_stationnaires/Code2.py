import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Paramètres
a = 1.0          # largeur totale du puits
V0 = 50.0        # profondeur du puits
L = 5.0          # taille du domaine spatial (de -L/2 à L/2)
N = 1000         # nombre de points de discrétisation

# Discrétisation de l'espace (de -L/2 à L/2)
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]

# Potentiel : puits centré en 0, de largeur a et profondeur -V0
V = np.zeros_like(x)
V[np.abs(x) <= a/2] = -V0

# Hamiltonien (différences finies)
kin = -2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)
kin /= dx**2
H = -kin + np.diag(V)  # H = T + V

# Diagonalisation (valeurs propres et fonctions d'onde)
E, psi = eigh(H)

# Tracé des états stationnaires
plt.figure(figsize=(12, 8))

# Affichage des 5 premiers états liés (E < 0)
n_affichage = 5
for n in range(n_affichage):
    psi_n = psi[:, n]
    psi_n /= np.sqrt(np.sum(psi_n**2) * dx)  # normalisation
    plt.plot(x, psi_n + E[n], label=f'n={n}, E={E[n]:.2f}')

plt.plot(x, V, 'k--', label='V(x)')  # potentiel
plt.title('États stationnaires du puits fini (centré en 0)')
plt.xlabel('x')
plt.ylabel('Énergie')
plt.legend()
plt.grid()
plt.show()
