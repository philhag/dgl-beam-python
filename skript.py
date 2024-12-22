# ## Differentialgleichung der Biegelinie am Einfeldträger
# Dieses Jupyter Notebook beschreibt die Differentialgleichung der Biegelinie am Einfeldträger
# und stellt Beispielrechnungen bereit.

# ### Import der Bibliotheken
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ### Definition der Symbole und Variablen
x, L, E, I, q0 = sp.symbols('x L E I q0')

# ### Differentialgleichung der Biegelinie
# Die allgemeine Differentialgleichung lautet:
# \[ E \cdot I \cdot \frac{d^4w(x)}{dx^4} = q(x) \]

# Beispiel: Gleichmäßig verteilte Last q(x) = q0
q = q0

# Differentialgleichung
w = sp.Function('w')(x)  # Durchbiegung w(x)
equation = sp.Eq(E * I * w.diff(x, 4), q)

# Ausgabe der Differentialgleichung
print("Differentialgleichung:")


# ### Lösung der Differentialgleichung
# Vierfache Integration der Differentialgleichung unter Berücksichtigung der Randbedingungen
# Randbedingungen für einen Einfeldträger (w(0) = 0, w(L) = 0, w'(0) = 0, w'(L) = 0)
C1, C2, C3, C4 = sp.symbols('C1 C2 C3 C4')
constants = [C1, C2, C3, C4]

# Vierfache Integration
w_solution = sp.integrate(q / (E * I), x, 4) + C1 * x**3 + C2 * x**2 + C3 * x + C4

# Randbedingungen anwenden
boundary_conditions = [
    sp.Eq(w_solution.subs(x, 0), 0),  # w(0) = 0
    sp.Eq(w_solution.subs(x, L), 0),  # w(L) = 0
    sp.Eq(w_solution.diff(x).subs(x, 0), 0),  # w'(0) = 0
    sp.Eq(w_solution.diff(x).subs(x, L), 0),  # w'(L) = 0
]

solutions = sp.solve(boundary_conditions, constants)

# Lösung der Durchbiegung
w_solution = w_solution.subs(solutions)
print("Lösung für die Durchbiegung w(x):")
display(w_solution)

# ### Ableitungen berechnen
# Biegemoment:
M = -E * I * sp.diff(w_solution, x, 2)
print("Biegemoment M(x):")
display(M)

# Querkraft:
V = sp.diff(M, x)
print("Querkraft V(x):")
display(V)

# Belastung:
q_calculated = sp.diff(V, x)
print("Belastung q(x):")
display(q_calculated)

# ### Visualisierung der Ergebnisse
# Parameterwerte festlegen
E_val = 2.1e11  # Elastizitätsmodul [Pa]
I_val = 8.333e-6  # Flächenträgheitsmoment [m^4]
q0_val = 1000  # Gleichmäßig verteilte Last [N/m]
L_val = 5  # Länge des Trägers [m]

# Funktionen numerisch auswerten
w_lambdified = sp.lambdify(x, w_solution.subs({E: E_val, I: I_val, q0: q0_val, L: L_val}), 'numpy')
M_lambdified = sp.lambdify(x, M.subs({E: E_val, I: I_val, q0: q0_val, L: L_val}), 'numpy')
V_lambdified = sp.lambdify(x, V.subs({E: E_val, I: I_val, q0: q0_val, L: L_val}), 'numpy')

# x-Werte generieren
x_vals = np.linspace(0, L_val, 500)
w_vals = w_lambdified(x_vals)
M_vals = M_lambdified(x_vals)
V_vals = V_lambdified(x_vals)

# Plot der Ergebnisse
plt.figure(figsize=(12, 8))

# Durchbiegung
plt.subplot(3, 1, 1)
plt.plot(x_vals, w_vals, label='Durchbiegung w(x)', color='blue')
plt.title('Durchbiegung, Biegemoment und Querkraft')
plt.ylabel('w(x) [m]')
plt.grid(True)
plt.legend()

# Biegemoment
plt.subplot(3, 1, 2)
plt.plot(x_vals, M_vals, label='Biegemoment M(x)', color='green')
plt.ylabel('M(x) [Nm]')
plt.grid(True)
plt.legend()

# Querkraft
plt.subplot(3, 1, 3)
plt.plot(x_vals, V_vals, label='Querkraft V(x)', color='red')
plt.xlabel('x [m]')
plt.ylabel('V(x) [N]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
