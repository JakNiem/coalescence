import pandas as pd
import numpy as np



#%% Convert values from the reduced system into the SI system (kg, meter, second, Kelvin, Joule, Pascal, etc.)
def reduced2SI(valueRed, quantity, fluid = 'Argon'):
    
    kb = 1.380649E-23 # J/K
    u = 1.660539E-27  # kg
    Na = 6.02214E+23  # 1/mol
    angstrom = 1e-10  # meter
    
    if fluid == 'Argon':  # Truncated and shifted
        sigma   = 3.3916  # Angstrom
        epsilon = 137.90  # K
        molmass = 39.948  # u = g/mol
        
    if fluid == 'Argon-like':  # Ge2007
        sigma   = 3.42    # Angstrom
        epsilon = 124     # K
        molmass = 39.987  # u = g/mol
        
    elif (fluid == 'LJTS') or (fluid == 'LJ'):
        sigma   = 1.0  # Angstrom
        epsilon = 1.0  # K
        molmass = 1.0  # u
    
    length = sigma*angstrom  # sigma in meter
    energy = epsilon*kb      # epsilon in J
    mass = molmass*u         # mass in kg
    
    time = length*((mass/energy)**0.5)  # second
    temperature = energy/kb  # K
    mol = 1/Na  # mol
    
    valueSI = valueRed
    
    # Thermal conductivity
    if quantity == 'thermal_cond':
        # Unit of energy/(time*length*temperature)
        valueSI *= energy/(time*length*temperature)
    
    # Shear viscosity (dynamic)
    elif quantity == 'shear_visco':
        # Unit of mass/(time*length)
        valueSI *= energy/(time*length)
    
    # Temperature
    elif (quantity == 'T') or (quantity.startswith('T_')):
        # Unit of energy/kb
        valueSI *= energy/kb
    
    # Mass specific energy
    elif (quantity == 'epot_mass') or (quantity == 'ekin_mass') or (quantity == 'h_mass') or (quantity == 'mu_mass'):
        # Unit of energy/mass
        valueSI *= energy/mass
    
    # Molar specific energy
    elif (quantity == 'epot') or (quantity == 'ekin') or (quantity == 'h') or (quantity == 'mu'):
        # Unit of energy/mol
        valueSI *= energy/mol
    
    # Mass specific density
    elif quantity == 'rho_mass':
        # Unit of mass/length^3
        valueSI *= mass/(length**3)
    
    # Molar specific density
    elif quantity == 'rho':
        # Unit of mol/length^3
        valueSI *= mol/(length**3)
    
    # Pressure
    elif (quantity == 'p') or (quantity.startswith('p_')):
        # Unit of energy/length^3
        valueSI *= energy/(length**3)
    
    # Number of particles
    elif (quantity == 'numParts') or (quantity == 'numSamples'):
        # Unit of mol
        valueSI *= mol
    
    # Position / Size / Length
    elif quantity == 'pos':
        # Unit of length
        valueSI *= length
    
    # Chemical potential
    elif quantity == 'chemPot_res':
        # Unit of energy/(mass*temperature)
        valueSI *= energy/(mass*temperature)
    
    # Energy flux
    elif (quantity == 'je') or (quantity == 'q') or (quantity.startswith('jEF_')) or (quantity.startswith('q_')):
        # Unit of energy/(length^2 time) or mass/(time^3)
        valueSI *= energy/((length**2)*time)
        # valueSI *= (mass/(time**3))  # alternative
    
    # Mass flux
    elif quantity == 'jp_mass':
        # Unit of mass/(length^2 time)
        valueSI *= mass/((length**2)*time)
        
    # Molar flux
    elif quantity == 'jp':
        # Unit of mol/(length^2 time)
        valueSI *= mol/((length**2)*time)
    
    # Velocity
    elif (quantity == 'v_x') or (quantity == 'v_y') or (quantity == 'v_z'):
        # Unit of length/time
        valueSI *= length/time
    
    # Force
    elif quantity.startswith('F_'):
        # Unit of energy/length (=Newton)
        valueSI *= energy/length
    
    # Surface tension
    elif quantity == 'gamma':
        # Unit of energy/length^2 (=Newton/m = kg/s^2)
        valueSI *= energy/(length**2)
    
    # Gas constant
    elif quantity == 'R':
        # Unit of energy/(mass*temperature) ; Specific gas constant
        valueSI *= energy/(mass*temperature)
    
    # Higher moment m_***
    elif quantity.startswith('m_'):
        # Unit of mass/(time^3)
        valueSI *= mass/(time**3)
    
    # Higher moments
    elif (quantity == 'delta') or (quantity == 'delta_res') or (quantity.startswith('R_')):
        # Unit of (mass*length)/(time^4)
        valueSI *= (mass*length)/(time**4)
    
    # Mass specific interface resistivity R11 / Ruu
    elif quantity == 'R11_mass':
        # Unit of (length^4)/(mass*temperature*time)
        valueSI *= (length**4)/(mass*temperature*time)
    
    # Molar specific interface resistivity R11 / Ruu
    elif quantity == 'R11':
        # Unit of (length^4)/(mol*temperature*time)
        valueSI *= (length**4)/(mol*temperature*time)                # (m^4)/(mol K s)
    
    # Mixed specific interface resistivity R11 / Ruu
    elif quantity == 'R11_mixed':
        # Unit of (length^4)/(mol*temperature*time)
        valueSI *= (energy*(length**2)*time)/((mol**2)*temperature)  # (J m^2 s)/(mol^2 K)
    
    # Mass specific interface resistivity R12 / Ruq or R21 / Rqu
    elif (quantity == 'R12_mass') or (quantity == 'R21_mass'):
        # Unit of (length^2 time)/(mass*temperature)
        valueSI *= ((length**2)*time)/(mass*temperature)
    
    # Molar specific interface resistivity R12 / Ruq or R21 / Rqu
    elif (quantity == 'R12') or (quantity == 'R21'):
        # Unit of (length^2 time)/(mol*temperature)
        valueSI *= ((length**2)*time)/(mol*temperature)
    
    # Interface resistivity R22 / Rqq
    elif quantity == 'R22':
        # Unit of (length^2 time)/(energy*temperature)
        valueSI *= ((length**2)*time)/(energy*temperature)
    
    else:
        valueSI = np.nan
        print(f'ERROR! Quantity {quantity} unknown')
    
    return valueSI



if __name__ == '__main__':
    print('Testing conversion')
    
    rho_red = 0.3
    rho_SI = reduced2SI(rho_red, 'rho', 'Argon')
    
    print(f'Density of Argon is: {rho_SI} mol/m^3')
    
