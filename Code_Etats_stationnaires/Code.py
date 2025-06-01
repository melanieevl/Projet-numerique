import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh  # diagonalisation

# Paramètres
a = 1.0       # demi-largeur du puits
V0 = 50.0     # hauteur du potentiel à l’extérieur
L = 5.0       # taille de la boîte numérique
N = 1000      # nombre de points
x = np.linspace(-L, L, N) # vecteur x allant de -L à +L avec N points
dx = x[1] - x[0] # pas d’espace entre deux points de x

# Potentiel
V = np.zeros_like(x)  # potentiel initialement nul partout
V[np.abs(x) > a] = V0 # en dehors de [-a, a], V = V0

# Hamiltonien : cinétique + potentiel H=T+V
# Partie cinétique (Laplacien discret 1D avec conditions aux bords de Dirichlet)
# Approximation : d²ψ/dx² ≈ [ψ(i+1) - 2ψ(i) + ψ(i-1)] / dx²
kin = -2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)  # matrice tridiagonale
kin /= dx**2
# Hamiltonien total H = T + V
H = -kin + np.diag(V) # ajout du potentiel sur la diagonale
# Attention : le signe - devant "kin" car T = -(ħ²/2m) * d²/dx² et ici ħ = m = 1 (unités réduites)

# Résolution numérique : diagonalisation

# On résout H ψ = E ψ pour obtenir :
# - E : énergies propres (valeurs propres)
# - psi : fonctions propres associées (vecteurs propres)
E, psi = eigh(H)# 'eigh' est utilisé car H est hermitien (réel symétrique ici)


# Visualisation des états stationnaires
# Tracer les 3 premiers états liés
plt.figure(figsize=(10, 6))  # taille du graphique


# On trace les 3 premiers états propres (les plus bas en énergie)
for n in range(3):
    psi_n = psi[:, n]# sélection de la n-ième fonction propre
    # Normalisation de la fonction d’onde pour que ∫|ψ|² dx = 1
    psi_n /= np.sqrt(np.sum(psi_n**2) * dx)
     # On décale la fonction d’onde verticalement de E[n] pour bien visualiser son niveau d’énergie
    plt.plot(x, psi_n + E[n], label=f'n={n}, E={E[n]:.2f}')

# Tracé du potentiel pour référence
plt.plot(x, V, 'k--', label='V(x)')

# Mise en forme du graphique
plt.title('États stationnaires du puits fini')
plt.xlabel('x') # axe des abscisses = position
plt.ylabel('Énergie') # axe des ordonnées = énergie
plt.legend() # affiche la légende (niveaux et potentiel)
plt.grid()  # ajoute une grille
plt.show() # affiche le tout
