import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh

# Define the input matrix
M = np.array([
    [0.25, 0, 0, 0.33],
    [0, 0.18, 0.1, 0],
    [0, 1, 0.12, 0],
    [0.17, -0.1, 0, 0.45]
])

# 1. Trace Test (Unit Trace)
matrix_trace = np.trace(M)
print(f"Trace of the matrix: {matrix_trace:.2f}")
if np.isclose(matrix_trace, 1.0):
    print("Trace Test Passed: The matrix has a unit trace.")
else:
    print("Trace Test Failed: The matrix does not have a unit trace.")

# 2. Hermiticity Test
is_hermitian = np.array_equal(M, M.T.conj())
if is_hermitian:
    print("Hermiticity Test Passed: The matrix is Hermitian.")
else:
    print("Hermiticity Test Failed: The matrix is not Hermitian.")

# 3. Positive Semi-Definiteness Test
eigenvals = np.linalg.eigvalsh(M)
print("Eigenvalues of the matrix:", eigenvals)
if np.all(eigenvals >= 0):
    print("Positive Semi-Definiteness Test Passed: All eigenvalues are non-negative.")
else:
    print("Positive Semi-Definiteness Test Failed: The matrix has negative eigenvalues.")

# 4. Purity Test
M_squared = np.dot(M, M)
purity_value = np.trace(M_squared)
print(f"Purity (Trace of Squared Matrix): {purity_value:.2f}")
if np.isclose(purity_value, 1.0):
    print("Purity Test Passed: The state is pure.")
else:
    print("Purity Test Failed: The state is mixed (Purity < 1).")

# Concurrence Calculation
sigma_y = np.array([
    [0, -1j],
    [1j, 0]
])
sigma_y_kron = np.kron(sigma_y, sigma_y)
R_matrix = np.dot(np.dot(M, sigma_y_kron), np.dot(M.conj(), sigma_y_kron))
R_eigenvalues, _ = eigh(R_matrix)
lambda_values = np.sqrt(np.abs(np.sort(R_eigenvalues)[::-1]))
concurrence_value = max(0, lambda_values[0] - lambda_values[1] - lambda_values[2] - lambda_values[3])
print("Concurrence:", concurrence_value)

# Visualizing the Density Matrix
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for the plot
X, Y = np.meshgrid(np.arange(M.shape[0]), np.arange(M.shape[1]))
X = X.flatten()
Y = Y.flatten()
Z = np.zeros_like(X)
height = M.flatten()

# Color coding based on values in the matrix
colors = ['Red' if val > 0.05 else 'Brown' if val < 0 else 'Blue' for val in M.flatten()]

# 3D Bar Plot
ax.bar3d(X, Y, Z, 0.5, 0.5, dz=height, color=colors, zsort='average')
ax.set_xticks(np.arange(M.shape[0]) + 0.25)
ax.set_yticks(np.arange(M.shape[1]) + 0.25)
ax.set_xticklabels(['|HH⟩', '|HV⟩', '|VH⟩', '|VV⟩'])
ax.set_yticklabels(['|HH⟩', '|HV⟩', '|VH⟩', '|VV⟩'])
ax.set_zlim(-0.2, 0.5)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Values')

# Title and Display
plt.title('Density Matrix Visualization for $|\\phi_{+}\\rangle$ State')
plt.show()
