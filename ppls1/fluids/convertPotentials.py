import pandas as pd
import numpy as np



#%% Convert values from the LJTS system into the Sutherland system
def LJTS2Sutherland(valueLJTS, temperature_LJTS, quantity):
    
    temperature_Sutherland = temperature_LJTS/1.49580967
    
    epsilon = 1.49580967  # energy
    sigma = 1.08161672 - 0.12344852*temperature_LJTS  # length
    mass = 0.82866477  # mass
    time = sigma*((mass/epsilon)**0.5)  # time
    
    if quantity == 'thermal_cond':
        valueSutherland = (valueLJTS*sigma**2)/((epsilon/mass)**0.5)
        
    elif quantity == 'shear_visco':
        valueSutherland = (valueLJTS*sigma**2)/((mass*epsilon)**0.5)
        
    elif (quantity == 'T') or (quantity.startswith('T_')):  # Unit of energy/k_b
        valueSutherland = valueLJTS/epsilon
        
    elif (quantity == 'epot') or (quantity == 'ekin'):  # Unit of energy/mass
        valueSutherland  = valueLJTS/(epsilon/mass)
        
    elif quantity == 'rho':  # Unit of 1/length^3
        valueSutherland = valueLJTS/(1/sigma**3)
        
    elif quantity == 'rho_mass':  # Unit of mass/length^3
        valueSutherland = valueLJTS/(mass/sigma**3)
        
    elif (quantity == 'p') or (quantity.startswith('p_')):  # Unit of energy/length^3
        valueSutherland = (valueLJTS*sigma**3)/epsilon
        
    elif (quantity == 'numParts') or (quantity == 'numSamples'):  # Unit of 1
        valueSutherland = valueLJTS
    
    elif quantity == 'pos':  # Unit of length
        valueSutherland = round(valueLJTS/sigma,6)
        
    elif (quantity == 'h') or (quantity == 'mu'):  # Unit of energy/mass
        valueSutherland  = valueLJTS/(epsilon/mass)
        
    elif quantity == 'chemPot_res':  # Unit of energy/(mass temperature)  TODO!!!!! ###############
        valueSutherland  = valueLJTS/(1/mass)
        
    elif (quantity == 'je') or (quantity == 'q') or (quantity.startswith('jEF_')) or (quantity.startswith('q_')):  # Unit of energy/(length^2 time) or mass/(time^3)
        valueSutherland = valueLJTS/(epsilon/((sigma**2)*time))
        # valueSutherland = valueLJTS/(mass/(time**3))  # alternative
        
    elif quantity == 'jp':  # Unit of mass/(length^2 time)
        valueSutherland = valueLJTS/(mass/((sigma**2)*time))
        
    elif (quantity == 'v_x') or (quantity == 'v_y') or (quantity == 'v_z'):  # Unit of length/time
        valueSutherland = valueLJTS/(sigma/time)
        
    elif quantity.startswith('F_'):  # Unit of epsilon/sigma (=Newton)
        valueSutherland = valueLJTS/(epsilon/sigma)
        
    elif quantity == 'R':  # Unit of energy/(mass temperature) ; Gas constant
        valueSutherland = valueLJTS/(epsilon/(mass*epsilon))
        
    elif quantity.startswith('m_'):  # Unit of mass/(time^3)
        valueSutherland = valueLJTS/(mass/(time**3))
        
    elif (quantity == 'delta') or (quantity == 'delta_res') or (quantity.startswith('R_')):  # Unit of (mass*sigma)/(time^4)
        valueSutherland = valueLJTS/((sigma)/(time**4))
        
    else:
        valueSutherland = valueLJTS
        print(f'ERROR! Quantity {quantity} unknown')
    
    return valueSutherland



if __name__ == '__main__':
    print('Testing conversion')
    
    T_LJTS = 1.08
    T_Sutherland = LJTS2Sutherland(T_LJTS,T_LJTS,'T')
    
    print(f'Critical temperature of Sutherland pot. is {T_Sutherland}')
    
