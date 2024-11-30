import numpy as np

# Define constants and input parameters
def material_properties(num_lamina, E1, E2, G12, nu12):
    """
    Generate material property matrices for the laminate.

    Parameters:
        num_lamina (int): Number of lamina in the laminate
        E1, E2 (list): Longitudinal and transverse modulus of elasticity for each lamina
        G12 (list): Shear modulus for each lamina
        nu12 (list): Poisson's ratio for each lamina

    Returns:
        Q (list): Stiffness matrix of each lamina
    """
    Q = []
    for i in range(num_lamina):
        q11 = E1[i] / (1 - nu12[i] * (E2[i] / E1[i]))
        q22 = E2[i] / (1 - nu12[i] * (E2[i] / E1[i]))
        q12 = nu12[i] * q22
        q66 = G12[i]
        Q.append(np.array([[q11, q12, 0],
                           [q12, q22, 0],
                           [0, 0, q66]]))
    return Q

def transform_stiffness(Q, theta):
    """
    Transform stiffness matrix for a given ply orientation.

    Parameters:
        Q (numpy array): Stiffness matrix of a lamina in the material coordinates
        theta (float): Lamina orientation in degrees

    Returns:
        Q_bar (numpy array): Transformed stiffness matrix
    """
    theta_rad = np.radians(theta)
    m = np.cos(theta_rad)
    n = np.sin(theta_rad)
    T = np.array([[m**2, n**2, 2*m*n],
                  [n**2, m**2, -2*m*n],
                  [-m*n, m*n, m**2 - n**2]])
    return np.linalg.inv(T) @ Q @ np.linalg.inv(T.T)

def assemble_laminate_properties(Q_bar, thicknesses):
    """
    Assemble the laminate stiffness matrices A, B, and D.

    Parameters:
        Q_bar (list): Transformed stiffness matrices of all lamina
        thicknesses (list): Thickness of each lamina

    Returns:
        A, B, D (numpy arrays): Laminate stiffness matrices
    """
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    z = np.cumsum([0] + thicknesses)  # Compute z-coordinates for each ply

    for i in range(len(thicknesses)):
        A += Q_bar[i] * (z[i + 1] - z[i])
        B += Q_bar[i] * (z[i + 1]**2 - z[i]**2) / 2
        D += Q_bar[i] * (z[i + 1]**3 - z[i]**3) / 3

    return A, B, D

def calculate_strain_and_stress(A, B, D, Nx, Ny, Nxy, Mx, My, Mxy, Q_bar, thicknesses):
    """
    Calculate strains, curvatures, and stresses in each lamina.

    Parameters:
        A, B, D (numpy arrays): Laminate stiffness matrices
        Nx, Ny, Nxy, Mx, My, Mxy (floats): In-plane and moment resultants
        Q_bar (list): Transformed stiffness matrices of all lamina
        thicknesses (list): Thickness of each lamina

    Returns:
        strain_lamina (list): Strains in material coordinates for each lamina
        stress_lamina (list): Stresses in material coordinates for each lamina
    """
    resultants = np.array([Nx, Ny, Nxy, Mx, My, Mxy])
    ABD = np.block([[A, B], [B, D]])
    midplane_strain_curvature = np.linalg.inv(ABD) @ resultants

    midplane_strain = midplane_strain_curvature[:3]
    curvature = midplane_strain_curvature[3:]

    z = np.cumsum([0] + thicknesses)  # Compute z-coordinates for each ply
    strain_lamina = []
    stress_lamina = []

    for i in range(len(thicknesses)):
        z_mid = (z[i + 1] + z[i]) / 2
        strain_global = midplane_strain + curvature * z_mid
        stress_global = Q_bar[i] @ strain_global

        strain_lamina.append(strain_global)
        stress_lamina.append(stress_global)

    return strain_lamina, stress_lamina

# Example Usage
num_lamina = 3
E1 = [150e9, 150e9, 150e9]  # Longitudinal modulus (Pa)
E2 = [10e9, 10e9, 10e9]  # Transverse modulus (Pa)
G12 = [5e9, 5e9, 5e9]  # Shear modulus (Pa)
nu12 = [0.25, 0.25, 0.25]  # Poisson's ratio
thicknesses = [0.125e-3, 0.125e-3, 0.125e-3]  # Ply thicknesses (m)
orients = [0, 45, -45]  # Ply orientations (degrees)

Q = material_properties(num_lamina, E1, E2, G12, nu12)
Q_bar = [transform_stiffness(Q[i], orients[i]) for i in range(num_lamina)]
A, B, D = assemble_laminate_properties(Q_bar, thicknesses)

Nx, Ny, Nxy = 1e3, 1e3, 0  # In-plane forces (N/m)
Mx, My, Mxy = 0, 0, 0  # Moments (N/m)

strain_lamina, stress_lamina = calculate_strain_and_stress(A, B, D, Nx, Ny, Nxy, Mx, My, Mxy, Q_bar, thicknesses)

print("Strains in each lamina:", strain_lamina)
print("Stresses in each lamina:", stress_lamina)
