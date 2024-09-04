import numpy as np

lamina_properties = { # Material properties of each lamina
    'E1':100e9,  # Longitudnal Young's modulus in Pa
    'E2':7e9,    # Transverse Young's modulud in Pa
    'G12':5e9,   # Shear modulus in Pa
    'v12':0.3    # Poisson's ratio
}

#Applied loads
loads = {
    'Nx':1300,  # Normal stress in x-direction in Pa
    'Ny':370,   # Normal stress in y-direction in Pa
    'Nxy':150,  # Shear stress in xy-direction in Pa
    'Mx':1700,  # Moment in x-direction in Nm
    'My':750,   # Moment in y-direction in Nm
    'Mxy':50    # Moment in xy-direction in Nm
}

stacking_sequence=[45,0,-45,90]  # Stacking sequence of the laminate

def deg_to_rad(deg):
    return deg*np.pi/180.0
def calculate_lamina_properties(theta):
    E1=lamina_properties['E1']
    E2=lamina_properties['E2']
    G12=lamina_properties['G12']
    v12=lamina_properties['v12']
    theta_rad=deg_to_rad(theta)
    
    Q11=E1/(1-v12*v12*E2/E1)
    Q22=E2/(1-v12*v12*E1/E2)
    Q12=v12*E2/(1-v12*v12*E1/E2)
    Q66=G12

    # Transformation matrix
    T = np.array([[np.cos(theta_rad)**2,np.sin(theta_rad)**2,2*np.sin(theta_rad)*np.cos(theta_rad)],
                  [np.sin(theta_rad)**2,np.cos(theta_rad)**2,-2*np.sin(theta_rad)*np.cos(theta_rad)],
                  [-np.sin(theta_rad)*np.cos(theta_rad), np.sin(theta_rad)*np.cos(theta_rad), 
                   np.cos(theta_rad)**2-np.sin(theta_rad)**2]])
    
    # Material properties in the laminate coordinate system
    Q_bar=np.dot(np.dot(np.linalg.inv(T),np.array([[Q11,Q12,0],[Q12,Q22,0],[0,0,Q66]])),T)
    return Q_bar

# Calculating stresses in the laminate
def calculate_stresses():
    print("Stresses in each lamina:")
    # Computing stress and strain 
    for i, theta in enumerate(stacking_sequence):
        Q_bar=calculate_lamina_properties(theta)
       
        print(f"Lamina {i+1},{loads['Nx']} Pa, {loads['Ny']} Pa, {loads['Nxy']} Pa")

def main():
    calculate_stresses()

if __name__ == "__main__":
    main()