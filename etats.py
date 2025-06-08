import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal import find_peaks

# === Paramètres du puits ===
V0 = -4000      # Potentiel du puits 
a = 0.5         # Largeur du puits
L = 10          # Taille du domaine de simulation
N = 2000        # Points pour discrétisation

# === Discrétisation de l’espace ===
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]

# === Potentiel carré ===
V = np.zeros_like(x)
V[np.abs(x) <= a/2] = V0

# === Hamiltonien (différences finies) ===
kin = -2*np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)
kin /= dx**2
H = -kin + np.diag(V)

# === Diagonalisation ===
E_vals, psi = eigh(H)

# === Calcul de la transmission analytique ===
E = np.linspace(0.1, 6000, 2000)
numerateur = V0**2
denominateur = 4 * E * (E - V0)
q = np.sqrt(E - V0)
sin_term = np.sin(q * a)**2
T_analytic = 1 / (1 + (numerateur / denominateur) * sin_term)

# === Détection des pics de résonance ===
peaks, _ = find_peaks(T_analytic, height=0.9)
resonance_energies = E[peaks]

# === Trouver les indices des états stationnaires proches des pics de résonance ===
indices_resonants = []
for Er in resonance_energies:
    idx = np.argmin(np.abs(E_vals - Er))
    if E_vals[idx] > 0:
        indices_resonants.append(idx)
indices_resonants = sorted(set(indices_resonants))

# === Plot 1 : n premiers états stationnaires ===
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
n_show = 20
amplification = 5  
for n in range(n_show):
    psi_n = psi[:, n]
    psi_n /= np.sqrt(np.sum(np.abs(psi_n)**2) * dx)
    plt.plot(x, amplification * psi_n + E_vals[n], label=f'n={n}, E={E_vals[n]:.1f}')
plt.plot(x, V, 'k--', label='V(x)')
plt.title("États stationnaires")
plt.xlim(-a, a)
plt.xlabel("x")
plt.ylim(-4500, 1000)
plt.ylabel("ψ(x) + E")
plt.grid(True)

# === Plot 2 : États stationnaires proches des pics de résonance ===
plt.subplot(1, 3, 2)
for idx in indices_resonants[:5]:  
    psi_n = psi[:, idx]
    psi_n /= np.sqrt(np.sum(np.abs(psi_n)**2) * dx)
    plt.plot(x, psi_n + E_vals[idx], label=f'n={idx}, E={E_vals[idx]:.1f}')
plt.plot(x, V, 'k--', label='V(x)')
plt.title("États respectant la condition de résonance")
plt.xlim(-a, a)
plt.xlabel("x")
plt.ylabel("ψ(x) + E")
plt.legend(fontsize=8)
plt.grid(True)

# === Plot 3 : Transmission ===
plt.subplot(1, 3, 3)
plt.plot(E, T_analytic, 'b', label="Analytique")
plt.legend()


def k_out(E): return np.sqrt(E)
def k_in(E): return np.sqrt(E - V0)

def M_segment(k2, L):
    delta = k2 * L
    return np.array([
        [np.cos(delta), 1j/k2 * np.sin(delta)],
        [1j * k2 * np.sin(delta), np.cos(delta)]
    ])

def transmission(E):
    k1 = k_out(E)
    k2 = k_in(E)
    if E - V0 < 0: return 0.0
    I1 = 0.5 * np.array([[1 + k2/k1, 1 - k2/k1],
                         [1 - k2/k1, 1 + k2/k1]])
    I2 = 0.5 * np.array([[1 + k1/k2, 1 - k1/k2],
                         [1 - k1/k2, 1 + k1/k2]])
    M = M_segment(k2, a)
    Mtot = np.linalg.inv(I1) @ M @ np.linalg.inv(I2)
    t = 1 / Mtot[0, 0]
    return np.abs(t)**2

T_matrix = np.array([transmission(e) for e in E])

plt.subplot(1, 3, 3)
plt.plot(E, T_matrix, 'r', label="Matrice de transfert")
plt.title("Coefficient de transmission par rapport à l'énergie")
plt.xlabel("Énergie E")
plt.ylabel("T(E)")
plt.grid(True)
plt.ylim(0, 1.1)
for Er in resonance_energies:
    plt.subplot(1, 3, 3)
    plt.axvline(Er, color='g', linestyle='--', alpha=0.7)
    plt.text(Er, 1.05, f"E={Er:.1f}", rotation=0, horizontalalignment='center', verticalalignment='bottom', color='g', fontsize=8)

plt.legend(loc='lower center', fontsize=8)


plt.tight_layout()
plt.show()
