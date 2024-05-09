import math
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.optimize import fsolve

import ppls1.fluids.ljts.petspy.petspy as petspy
from ppls1.fluids.ljts.ljts import sat_vrabec2006
from ppls1.fluids.therm_cond import lambda_lauten


#%% Function to calculate interface resistivities from evaporation profiles
def direct_ifRes_evap(df, Ln=10, Lv=20, pos_x_if_l=None, pos_x_if_v=15, const_l=False, const_v=False, R=1.0):
    '''
    Function to calculate interface resistivities from evaporation profiles

    :param dataframe df: Dataframe with density, temperature and chem. pot. profile where interface is at position/index 0 (requires columns: rho, T, h, g, jp, je)
    :param float Ln: Length of non-thermostated bulk liquid
    :param float Lv: Length of bulk vapor
    :param float pos_x_if_l: Position of left boundary of interface
    :param float pos_x_if_v: Position of right boundary of interface
    :param bool const_l: Set if liquid should be treated as constant
    :param bool const_l: Set if vapor should be treated as constant
    :return: Dataframe including the interface resistivity data
    '''
    
    #%% Calculate aux. variables
    
    # Position of left boundary of interface
    if pos_x_if_l is None:
        x_if_l = df['rho'][-10:10].idxmax()  # Inflection point
    else:
        x_if_l = df.iloc[df.index.get_loc(pos_x_if_l, method='nearest')].name
        
    x_if_v = df.iloc[df.index.get_loc(pos_x_if_v, method='nearest')].name   # Position of right boundary of interface
    
    x_bulk_l = x_if_l-Ln
    x_bulk_v = x_if_v+Lv
    
    jp_const = round(df['jp'][x_if_v:x_bulk_v].mean(), 8)
    
    je_const = round(df['je'][x_if_v:x_bulk_v].mean(), 8)
    
    q_v_const = round(df['q'][x_if_v:x_bulk_v].mean(), 8)
    
    x_if = df.index[df.index>=x_if_l]
    x_if = x_if[x_if<=x_if_v]
    
    h_v = df['h'][x_if_v]
    
    T_v = df['T'][x_if_v]  # Temperature at right boundary of IF
    T_l = df['T'][x_if_l]  # Temperature at left boundary of IF
    
    mu_v = df['mu'][x_if_v]
    mu_l = df['mu'][x_if_l]
    
    T_liq = df.iloc[0]['T']  # At index = 0
    
    T_if = df['T'][0.0]  # Interface temperature (at position = 0)
    
    rho_v = df['rho'][x_if_v]
    
    R = R  # 1.0 for LJTS and 0.82866477 for Sutherland
    
    
    #%% Direct calculation
    
    Xq_v = (1/T_v) - (1/T_l)
    Xu_v = ((mu_l/T_l) - (mu_v/T_v)) + h_v * ((1/T_v) - (1/T_l))
    
    R_22_v_direct = np.nan           # r_qq
    R_21_v_direct = Xq_v / jp_const  # r_qu
    R_12_v_direct = np.nan           # r_uq
    R_11_v_direct = Xu_v / jp_const  # r_uu
    
    # Dimensionless according to Henning (Eq. 31 & 32)
    fac_dimless_r_11 =                    (rho_v/R)*np.sqrt((R*T_if)/(2*math.pi))
    fac_dimless_r_12 = fac_dimless_r_21 = (rho_v*T_if)*np.sqrt((R*T_if)/(2*math.pi))
    fac_dimless_r_22 =                    (rho_v*R*(T_if**2))*np.sqrt((R*T_if)/(2*math.pi))
    
    
    R_22_v_direct_dimless = np.nan             # r_qq_dach
    R_21_v_direct_dimless = (fac_dimless_r_21*Xq_v) / jp_const  # r_qu_dach
    R_12_v_direct_dimless = np.nan             # r_uq_dach
    R_11_v_direct_dimless = (fac_dimless_r_11*Xu_v) / jp_const  # r_uu_dach
    
    
    # With extrapolation
    # from bulk liquid to interface (liquid)
    dfLiquid = df[x_bulk_l:x_if_l].copy()
    if const_l:
        dfLiquid['rho'] = dfLiquid['rho'].mean()
        dfLiquid['T']   = dfLiquid['T'].mean()
        dfLiquid['h']   = dfLiquid['h'].mean()
        dfLiquid['mu']  = dfLiquid['mu'].mean()
    
    rho_l_extrapol = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['rho'], 1))(0)
    T_l_extrapol   = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['T'], 1))(0)
    h_l_extrapol   = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['h'], 1))(0)
    mu_l_extrapol  = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['mu'], 1))(0)
    
    
    # from interface (vapor) to bulk vapor
    dfVapor = df[x_if_v:x_bulk_v].copy()
    if const_v:
        dfVapor['rho'] = dfVapor['rho'].mean()
        dfVapor['T']   = dfVapor['T'].mean()
        dfVapor['h']   = dfVapor['h'].mean()
        dfVapor['mu']  = dfVapor['mu'].mean()
    
    rho_v_extrapol = np.poly1d(np.polyfit(dfVapor.index, dfVapor['rho'], 1))(0)
    T_v_extrapol   = np.poly1d(np.polyfit(dfVapor.index, dfVapor['T'], 1))(0)
    h_v_extrapol   = np.poly1d(np.polyfit(dfVapor.index, dfVapor['h'], 1))(0)
    mu_v_extrapol  = np.poly1d(np.polyfit(dfVapor.index, dfVapor['mu'], 1))(0)
    
    
    Xq_v_extrapol = (1/T_v_extrapol) - (1/T_l_extrapol)
    Xu_v_extrapol = ((mu_l_extrapol/T_l_extrapol) - (mu_v_extrapol/T_v_extrapol)) + h_v_extrapol * ((1/T_v_extrapol) - (1/T_l_extrapol))
    
    R_22_v_direct_extrapol = np.nan           # r_qq
    R_21_v_direct_extrapol = Xq_v_extrapol / jp_const  # r_qu
    R_12_v_direct_extrapol = np.nan           # r_uq
    R_11_v_direct_extrapol = Xu_v_extrapol / jp_const  # r_uu
    
    # Dimensionless according to Henning (Eq. 31 & 32)
    fac_dimless_r_11_extrapol =                             (rho_v_extrapol/R)*np.sqrt((R*T_if)/(2*math.pi))
    fac_dimless_r_12_extrapol = fac_dimless_r_21_extrapol = (rho_v_extrapol*T_if)*np.sqrt((R*T_if)/(2*math.pi))
    fac_dimless_r_22_extrapol =                             (rho_v_extrapol*R*(T_if**2))*np.sqrt((R*T_if)/(2*math.pi))
    
    R_22_v_direct_dimless_extrapol = np.nan             # r_qq_dach
    R_21_v_direct_dimless_extrapol = (fac_dimless_r_21_extrapol*Xq_v_extrapol) / jp_const  # r_qu_dach
    R_12_v_direct_dimless_extrapol = np.nan             # r_uq_dach
    R_11_v_direct_dimless_extrapol = (fac_dimless_r_11_extrapol*Xu_v_extrapol) / jp_const  # r_uu_dach
    
    
    #%% Save in dataframe
    
    if 'v_y' not in df.columns:
        df['v_y'] = df['jp']/df['rho']
    
    v_z = df[x_if_v:x_bulk_v]['v_y'].mean()
    
    if 'p' in dfVapor.columns:
        p_v = dfVapor['p'].mean()
    else:
        p_v = np.nan
    
    T_v = np.poly1d(np.polyfit(dfVapor.index, dfVapor['T'], 1))(x_bulk_v)
    
    dfIF = pd.Series(dtype=np.float64)
    dfIF['T_liq'] = T_liq
    dfIF['v_z'] = v_z
    dfIF['p_v'] = p_v
    dfIF['T_if'] = T_if
    dfIF['T_v'] = T_v
    dfIF['je'] = je_const
    dfIF['jp'] = jp_const
    dfIF['q_v'] = q_v_const  # Should be 0.0
    dfIF['x_if_l'] = x_if_l
    dfIF['x_if_v'] = x_if_v
    dfIF['x_bulk_l'] = x_bulk_l
    dfIF['x_bulk_v'] = x_bulk_v
    dfIF['Xq_v'] = Xq_v
    dfIF['Xu_v'] = Xu_v
    dfIF['rho_l_extrapol'] = rho_l_extrapol
    dfIF['T_l_extrapol']   = T_l_extrapol
    dfIF['h_l_extrapol']   = h_l_extrapol
    dfIF['mu_l_extrapol']  = mu_l_extrapol
    dfIF['rho_v_extrapol'] = rho_v_extrapol
    dfIF['T_v_extrapol']   = T_v_extrapol
    dfIF['h_v_extrapol']   = h_v_extrapol
    dfIF['mu_v_extrapol']  = mu_v_extrapol
    dfIF['rho_l_bulk_fit'] = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['rho'], 1))(x_bulk_l)
    dfIF['T_l_bulk_fit']   = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['T'], 1))(x_bulk_l)
    dfIF['h_l_bulk_fit']   = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['h'], 1))(x_bulk_l)
    dfIF['mu_l_bulk_fit']  = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['mu'], 1))(x_bulk_l)
    dfIF['rho_v_bulk_fit'] = np.poly1d(np.polyfit(dfVapor.index, dfVapor['rho'], 1))(x_bulk_v)
    dfIF['T_v_bulk_fit']   = T_v
    dfIF['h_v_bulk_fit']   = np.poly1d(np.polyfit(dfVapor.index, dfVapor['h'], 1))(x_bulk_v)
    dfIF['mu_v_bulk_fit']  = np.poly1d(np.polyfit(dfVapor.index, dfVapor['mu'], 1))(x_bulk_v)
    dfIF['R_22_v_direct_dimless'] = R_22_v_direct_dimless # r_qq
    dfIF['R_12_v_direct_dimless'] = R_12_v_direct_dimless # r_uq
    dfIF['R_21_v_direct_dimless'] = R_21_v_direct_dimless # r_qu
    dfIF['R_11_v_direct_dimless'] = R_11_v_direct_dimless # r_uu
    dfIF['R_22_v_direct'] = R_22_v_direct # r_qq
    dfIF['R_12_v_direct'] = R_12_v_direct # r_uq
    dfIF['R_21_v_direct'] = R_21_v_direct # r_qu
    dfIF['R_11_v_direct'] = R_11_v_direct # r_uu
    dfIF['R_22_v_direct_dimless_extrapol'] = R_22_v_direct_dimless_extrapol # r_qq
    dfIF['R_12_v_direct_dimless_extrapol'] = R_12_v_direct_dimless_extrapol # r_uq
    dfIF['R_21_v_direct_dimless_extrapol'] = R_21_v_direct_dimless_extrapol # r_qu
    dfIF['R_11_v_direct_dimless_extrapol'] = R_11_v_direct_dimless_extrapol # r_uu
    dfIF['R_22_v_direct_extrapol'] = R_22_v_direct_extrapol # r_qq
    dfIF['R_12_v_direct_extrapol'] = R_12_v_direct_extrapol # r_uq
    dfIF['R_21_v_direct_extrapol'] = R_21_v_direct_extrapol # r_qu
    dfIF['R_11_v_direct_extrapol'] = R_11_v_direct_extrapol # r_uu
    
    
    return dfIF



#%% Function to calculate interface resistivities from temperature gradient profiles
def direct_ifRes_tempGrad(df, Ln=10, Lv=50, pos_x_if_l=-7, pos_x_if_v=7, const_l=False, const_v=False, R=1.0):
    '''
    Function to calculate interface resistivities from temperature gradient

    :param dataframe df: Dataframe with density, temperature and chem. pot. profile where interface is at position/index 0 (requires columns: rho, T, h, g, jp, je)
    :param float Ln: Length of non-thermostated bulk liquid
    :param float Lv: Length of bulk vapor
    :param float pos_x_if_l: Position of left boundary of interface
    :param float pos_x_if_v: Position of right boundary of interface
    :param bool const_l: Set if liquid should be treated as constant
    :param bool const_l: Set if vapor should be treated as constant
    :return: Dataframe including the interface resistivity data
    '''
    
    #%% Calculate aux. variables
    
    x_if_l = df.iloc[df.index.get_loc(pos_x_if_l, method='nearest')].name   # Position of right boundary of interface
    x_if_v = df.iloc[df.index.get_loc(pos_x_if_v, method='nearest')].name   # Position of right boundary of interface
    
    x_bulk_l = x_if_l-Ln
    x_bulk_v = x_if_v+Lv
    
    x_if = df.index[df.index>=x_if_l]
    x_if = x_if[x_if<=x_if_v]
    
    q_v_const = round(df['q'][x_if_v:x_bulk_v].mean(), 8)
    
    jp_const = round(df['jp'][x_if_v:x_bulk_v].mean(), 8)
    
    je_const = round(df['je'][x_if_v:x_bulk_v].mean(), 8)
    
    h_v = df['h'][x_if_v]
    # h_l = df['h'][x_if_l]
    
    T_v = df['T'][x_if_v]  # Temperature at right boundary of IF
    T_l = df['T'][x_if_l]  # Temperature at left boundary of IF
    
    mu_v = df['mu'][x_if_v]
    mu_l = df['mu'][x_if_l]
    
    T_liq = df.iloc[0]['T']  # At index = 0
    
    T_if = df['T'][0.0]  # Interface temperature (at position = 0)
    
    rho_v = df['rho'][x_if_v]
    
    R = R  # 1.0 for LJTS and 0.82866477 for Sutherland
    
    
    #%% Direct calculation
    
    Xq_v = (1/T_v) - (1/T_l)
    Xu_v = ((mu_l/T_l) - (mu_v/T_v)) + h_v * ((1/T_v) - (1/T_l))
    
    R_22_v_direct = Xq_v / q_v_const  # r_qq
    R_21_v_direct = np.nan            # r_qu
    R_12_v_direct = Xu_v / q_v_const  # r_uq
    R_11_v_direct = np.nan            # r_uu
    
    # Dimensionless according to Henning (Eq. 31 & 32)
    fac_dimless_r_11 =                    (rho_v/R)*np.sqrt((R*T_if)/(2*math.pi))
    fac_dimless_r_12 = fac_dimless_r_21 = (rho_v*T_if)*np.sqrt((R*T_if)/(2*math.pi))
    fac_dimless_r_22 =                    (rho_v*R*(T_if**2))*np.sqrt((R*T_if)/(2*math.pi))
    
    
    R_22_v_direct_dimless = (fac_dimless_r_22*Xq_v) / q_v_const  # r_qq_dach
    R_21_v_direct_dimless = np.nan                               # r_qu_dach
    R_12_v_direct_dimless = (fac_dimless_r_12*Xu_v) / q_v_const  # r_uq_dach
    R_11_v_direct_dimless = np.nan                               # r_uu_dach
    
    
    # With extrapolation
    # from bulk liquid to interface (liquid)
    dfLiquid = df[x_bulk_l:x_if_l].copy()
    if const_l:
        dfLiquid['rho'] = dfLiquid['rho'].mean()
        dfLiquid['T']   = dfLiquid['T'].mean()
        dfLiquid['h']   = dfLiquid['h'].mean()
        dfLiquid['mu']  = dfLiquid['mu'].mean()
    
    rho_l_extrapol = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['rho'], 1))(0)
    T_l_extrapol   = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['T'], 1))(0)
    h_l_extrapol   = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['h'], 1))(0)
    mu_l_extrapol  = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['mu'], 1))(0)
    
    
    # from interface (vapor) to bulk vapor
    dfVapor = df[x_if_v:x_bulk_v].copy()
    if const_v:
        dfVapor['rho'] = dfVapor['rho'].mean()
        dfVapor['T']   = dfVapor['T'].mean()
        dfVapor['h']   = dfVapor['h'].mean()
        dfVapor['mu']  = dfVapor['mu'].mean()
    
    rho_v_extrapol = np.poly1d(np.polyfit(dfVapor.index, dfVapor['rho'], 1))(0)
    T_v_extrapol   = np.poly1d(np.polyfit(dfVapor.index, dfVapor['T'], 1))(0)
    h_v_extrapol   = np.poly1d(np.polyfit(dfVapor.index, dfVapor['h'], 1))(0)
    mu_v_extrapol  = np.poly1d(np.polyfit(dfVapor.index, dfVapor['mu'], 1))(0)
    
    
    Xq_v_extrapol = (1/T_v_extrapol) - (1/T_l_extrapol)
    Xu_v_extrapol = ((mu_l_extrapol/T_l_extrapol) - (mu_v_extrapol/T_v_extrapol)) + h_v_extrapol * ((1/T_v_extrapol) - (1/T_l_extrapol))
    
    R_22_v_direct_extrapol = Xq_v_extrapol / q_v_const  # r_qq
    R_21_v_direct_extrapol = np.nan                     # r_qu
    R_12_v_direct_extrapol = Xu_v_extrapol / q_v_const  # r_uq
    R_11_v_direct_extrapol = np.nan                     # r_uu
    
    # Dimensionless according to Henning (Eq. 31 & 32)
    fac_dimless_r_11_extrapol =                             (rho_v_extrapol/R)*np.sqrt((R*T_if)/(2*math.pi))
    fac_dimless_r_12_extrapol = fac_dimless_r_21_extrapol = (rho_v_extrapol*T_if)*np.sqrt((R*T_if)/(2*math.pi))
    fac_dimless_r_22_extrapol =                             (rho_v_extrapol*R*(T_if**2))*np.sqrt((R*T_if)/(2*math.pi))
    
    R_22_v_direct_dimless_extrapol = (fac_dimless_r_22_extrapol*Xq_v_extrapol) / q_v_const  # r_qq_dach
    R_21_v_direct_dimless_extrapol = np.nan                                                 # r_qu_dach
    R_12_v_direct_dimless_extrapol = (fac_dimless_r_12_extrapol*Xu_v_extrapol) / q_v_const  # r_uq_dach
    R_11_v_direct_dimless_extrapol = np.nan                                                 # r_uu_dach
    
    
    #%% Save in dataframe
    
    if 'v_y' not in df.columns:
        df['v_y'] = df['jp']/df['rho']
    
    v_z = df[x_if_v:x_bulk_v]['v_y'].mean()
    
    T_v = df['T'][df.index[-30]:df.index[-10]].mean()
    # T_v = df['T'][90:110].mean()
    
    dT = ( T_v - df['T'][:x_bulk_l].mean() ) / ( df.index[-20] - x_bulk_l )  # Gradient between left and right end of domain
    
    if 'p' in dfVapor.columns:
        p_v = dfVapor['p'].mean()
    else:
        p_v = np.nan
    
    dfIF = pd.Series(dtype=np.float64)
    dfIF['T_liq'] = T_liq
    dfIF['v_z'] = v_z
    dfIF['dT'] = dT
    dfIF['p_v'] = p_v
    dfIF['T_if'] = T_if
    dfIF['T_v'] = T_v
    dfIF['je'] = je_const
    dfIF['jp'] = jp_const  # Should be 0.0
    dfIF['q_v'] = q_v_const
    dfIF['x_if_l'] = x_if_l
    dfIF['x_if_v'] = x_if_v
    dfIF['x_bulk_l'] = x_bulk_l
    dfIF['x_bulk_v'] = x_bulk_v
    dfIF['Xq_v'] = Xq_v
    dfIF['Xu_v'] = Xu_v
    dfIF['rho_l_extrapol'] = rho_l_extrapol
    dfIF['T_l_extrapol']   = T_l_extrapol
    dfIF['h_l_extrapol']   = h_l_extrapol
    dfIF['mu_l_extrapol']  = mu_l_extrapol
    dfIF['rho_v_extrapol'] = rho_v_extrapol
    dfIF['T_v_extrapol']   = T_v_extrapol
    dfIF['h_v_extrapol']   = h_v_extrapol
    dfIF['mu_v_extrapol']  = mu_v_extrapol
    dfIF['rho_l_bulk_fit'] = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['rho'], 1))(x_bulk_l)
    dfIF['T_l_bulk_fit']   = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['T'], 1))(x_bulk_l)
    dfIF['h_l_bulk_fit']   = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['h'], 1))(x_bulk_l)
    dfIF['mu_l_bulk_fit']  = np.poly1d(np.polyfit(dfLiquid.index, dfLiquid['mu'], 1))(x_bulk_l)
    dfIF['rho_v_bulk_fit'] = np.poly1d(np.polyfit(dfVapor.index, dfVapor['rho'], 1))(x_bulk_v)
    dfIF['T_v_bulk_fit']   = np.poly1d(np.polyfit(dfVapor.index, dfVapor['T'], 1))(x_bulk_v)
    dfIF['h_v_bulk_fit']   = np.poly1d(np.polyfit(dfVapor.index, dfVapor['h'], 1))(x_bulk_v)
    dfIF['mu_v_bulk_fit']  = np.poly1d(np.polyfit(dfVapor.index, dfVapor['mu'], 1))(x_bulk_v)
    dfIF['R_22_v_direct_dimless'] = R_22_v_direct_dimless # r_qq
    dfIF['R_12_v_direct_dimless'] = R_12_v_direct_dimless # r_uq
    dfIF['R_21_v_direct_dimless'] = R_21_v_direct_dimless # r_qu
    dfIF['R_11_v_direct_dimless'] = R_11_v_direct_dimless # r_uu
    dfIF['R_22_v_direct'] = R_22_v_direct # r_qq
    dfIF['R_12_v_direct'] = R_12_v_direct # r_uq
    dfIF['R_21_v_direct'] = R_21_v_direct # r_qu
    dfIF['R_11_v_direct'] = R_11_v_direct # r_uu
    dfIF['R_22_v_direct_dimless_extrapol'] = R_22_v_direct_dimless_extrapol # r_qq
    dfIF['R_12_v_direct_dimless_extrapol'] = R_12_v_direct_dimless_extrapol # r_uq
    dfIF['R_21_v_direct_dimless_extrapol'] = R_21_v_direct_dimless_extrapol # r_qu
    dfIF['R_11_v_direct_dimless_extrapol'] = R_11_v_direct_dimless_extrapol # r_uu
    dfIF['R_22_v_direct_extrapol'] = R_22_v_direct_extrapol # r_qq
    dfIF['R_12_v_direct_extrapol'] = R_12_v_direct_extrapol # r_uq
    dfIF['R_21_v_direct_extrapol'] = R_21_v_direct_extrapol # r_qu
    dfIF['R_11_v_direct_extrapol'] = R_11_v_direct_extrapol # r_uu
    
    
    return dfIF


#%% Function to calculate interface resistivities based on Integral Relations
def integralRel_ifRes(df, pos_x_if_l=None, pos_x_if_v=15, R=1.0):
    '''
    Function to calculate interface resistivities based on Integral Relations

    :param dataframe df: Dataframe with density, temperature and chem. pot. profile where interface is at position/index 0 (requires columns: rho, T, h, g, jp, je)
    :param float pos_x_if_l: Position of left boundary of interface
    :param float pos_x_if_v: Position of right boundary of interface
    :return: Dataframe including the interface resistivity data
    ''' 
    
    #%% Calculate aux. variables
    
    df['dT_inv'] = (1/df['T']).diff()/(df.index.to_series().diff())
    
    # Position of left boundary of interface
    if pos_x_if_l is None:
        x_if_l = df['rho'][-10:10].idxmax()  # Inflection point
    else:
        x_if_l = df.iloc[df.index.get_loc(pos_x_if_l, method='nearest')].name
    
    x_if_v = df.iloc[df.index.get_loc(pos_x_if_v, method='nearest')].name   # Position of right boundary of interface
    
    print(x_if_l,x_if_v)
    
    x_if = df[x_if_l:x_if_v].index
    
    h_vap = df['h'][x_if_v+50:x_if_v+100].mean()
    # h_liq = df['h'][x_if_l-15:x_if_l-11].mean()
    
    T_if = df['T'][0.0]  # At position = 0
    
    _,rho_vap,_ = sat_vrabec2006(T_if)
    
    R = R  # 1.0 for LJTS and 0.82866477 for Sutherland
    
    
    #%% Integral relations Johannessen2006
    df['r_22'] = df['dT_inv']/(df['q'])
    
    df['r_21_v'] = df['r_12_v'] = df['r_22']*(h_vap - df['h'])
    #df['r_21_l'] = df['r_12_l'] = df['r_22']*(h_liq - df['h'])
    
    df['r_11_v'] = df['r_22']*(h_vap - df['h'])**2
    #df['r_11_l'] = df['r_22']*(h_liq - df['h'])**2
    
    
    ### Clip pole to maximum value
    # maxClip_idx = 3.5
    # df.loc[df['r_22'] >= df['r_22'][-10:maxClip_idx].max(), 'r_22'] = df['r_22'][-10:maxClip_idx].max()
    # df.loc[df['r_21_v'] >= df['r_21_v'][-10:maxClip_idx].max(), 'r_21_v'] = df['r_21_v'][-10:maxClip_idx].max()
    # df.loc[df['r_12_v'] >= df['r_12_v'][-10:maxClip_idx].max(), 'r_12_v'] = df['r_12_v'][-10:maxClip_idx].max()
    # df.loc[df['r_11_v'] >= df['r_11_v'][-10:maxClip_idx].max(), 'r_11_v'] = df['r_11_v'][-10:maxClip_idx].max()
    
    
    R_22_v = integrate.trapz(df['r_22'][x_if_l:x_if_v], x=x_if)
    # R_22_l = integrate.trapz(df['r_22'][x_if_l:x_if_v], x=x_if)
    
    R_21_v = integrate.trapz(df['r_21_v'][x_if_l:x_if_v], x=x_if)
    R_12_v = integrate.trapz(df['r_12_v'][x_if_l:x_if_v], x=x_if)
    #R_21_l = integrate.trapz(df['r_21_l'][x_if_l:x_if_v], x=x_if)
    #R_12_l = integrate.trapz(df['r_12_l'][x_if_l:x_if_v], x=x_if)
    
    R_11_v = integrate.trapz(df['r_11_v'][x_if_l:x_if_v], x=x_if)
    #R_11_l = integrate.trapz(df['r_11_l'][x_if_l:x_if_v], x=x_if)
    
    
    # Dimensionless according to Henning (Eq. 31 & 32)
    fac_dimless_r_11 =                    (rho_vap/R)*np.sqrt((R*T_if)/(2*math.pi))
    fac_dimless_r_12 = fac_dimless_r_21 = (rho_vap*T_if)*np.sqrt((R*T_if)/(2*math.pi))
    fac_dimless_r_22 =                    (rho_vap*R*(T_if**2))*np.sqrt((R*T_if)/(2*math.pi))
    
    R_11_v_dimless = fac_dimless_r_11*R_11_v
    R_12_v_dimless = fac_dimless_r_12*R_12_v
    R_21_v_dimless = fac_dimless_r_21*R_21_v
    R_22_v_dimless = fac_dimless_r_22*R_22_v
    
    # Onsager coeff.
    try:
        Rg = [[R_22_v, R_21_v], [R_12_v, R_11_v]]
        Og = np.linalg.inv(Rg)
        
        if not np.allclose(np.dot(Rg, Og), np.eye(2)): print('Warning! Inv. of R_v not closed')
        
        O_22_v = Og[0][0]
        O_21_v = Og[0][1]
        O_12_v = Og[1][0]
        O_11_v = Og[1][1]
    except:
        O_22_v = O_21_v = O_12_v = O_11_v = np.nan
        
    
    
    #%% Save in dataframe
    
    dfIF = pd.Series(dtype=np.float64)
    dfIF['R_11_v_integral'] = R_11_v
    dfIF['R_12_v_integral'] = R_12_v
    dfIF['R_21_v_integral'] = R_21_v
    dfIF['R_22_v_integral'] = R_22_v
    dfIF['R_11_v_integral_dimless'] = R_11_v_dimless
    dfIF['R_12_v_integral_dimless'] = R_12_v_dimless
    dfIF['R_21_v_integral_dimless'] = R_21_v_dimless
    dfIF['R_22_v_integral_dimless'] = R_22_v_dimless
    dfIF['O_11_v_integral'] = O_11_v
    dfIF['O_12_v_integral'] = O_12_v
    dfIF['O_21_v_integral'] = O_21_v
    dfIF['O_22_v_integral'] = O_22_v
    
    return dfIF


#%% Function to calculate interface resistivities and other quantities analytically
def analSol_ifRes_evap(T_liq, v_z, L_n, j_p, j_e, printSolution=True):
    '''
    Function to calculate interface resistivities and other quantities analytically

    :param float T_liq: Liquid temperature
    :param float v_z: Hydrodynamic vapor velocity
    :param float L_n: Length of non-thermostated liquid
    :param float j_p: Mass flux
    :param float j_e: Energy flux
    :param bool printSolution: Print calculated values
    :return: List including the calculated data
    '''
    
    # Saturated liquid density based on Vrabec2006
    def rhol_vrabec2006(T):
        rc=0.3190
        dt=1.0779-T
        a,b,c=0.5649,0.1314,0.0413
        rhol=rc+a*dt**(1/3.)+b*dt+c*dt**(3/2.)
        return rhol
    
    # System of equations
    def equations(y):
        T_if, rho_if, mu_if, T_vap, h_vap, mu_vap, R_11, R_21 = y
        eqs = [
            # equation 1
            rho_if - rhol_vrabec2006(T_if),
            # equation 2
            mu_if - float(petspy.petseos(12,rho_if,19,T_if,51)),
            # equation 3
            h_vap - (float(petspy.petseos(12,rho_vap,19,T_vap,14))+2.0),  # Referenzpunkt
            # equation 4
            mu_vap - float(petspy.petseos(12,rho_vap,19,T_vap,51)),
            # equation 5
            R_21*j_p - ((1/T_vap)-(1/T_if)),
            # equation 6
            R_11*j_p - ( ((mu_if/T_if)-(mu_vap/T_vap)) + h_vap*((1/T_vap)-(1/T_if)) ),
            # equation 7
            j_e - ((h_vap + ((v_z**2)/2))*j_p),
            # equation 8
            j_e - (h_liq*j_p + lambda_liq*((T_liq-T_if)/L_n)),
        ]
        return eqs
    
    
    # Direkte Berechungen
    rho_liq,_,_ = sat_vrabec2006(T_liq)
    h_liq = float(petspy.petseos(12,rho_liq,19,T_liq,14))+2.0  # Referenzpunkt
    mu_liq = float(petspy.petseos(12,rho_liq,19,T_liq,51))
    lambda_liq = lambda_lauten(T_liq,rho_liq)
    rho_vap = j_p/v_z
    
    
    # Unbekannte und initial conditions: T_if, rho_if, mu_if, T_vap, h_vap, mu_vap, R_11, R_21
    T_if = 0.95*T_liq
    rho_if,_,_ = sat_vrabec2006(T_if)
    mu_if = float(petspy.petseos(12,rho_if,19,T_if,51))
    T_vap = 0.8*T_liq
    h_vap = float(petspy.petseos(12,rho_vap,19,T_vap,14))+2.0  # Referenzpunkt
    mu_vap = float(petspy.petseos(12,rho_vap,19,T_vap,51))
    
    R_11 = 100  # First guess
    R_21 = 10   # First guess
    
    y0 = [T_if, rho_if, mu_if, T_vap, h_vap, mu_vap, R_11, R_21]
    
    # Solve the system of equations
    sol = fsolve(equations, y0)
    T_if, rho_if, mu_if, T_vap, h_vap, mu_vap, R_11, R_21 = sol
    
    # Dimensionless according to Henning (Eq. 31 & 32)
    R = 1.0
    fac_dimless_r_11_extrapol = (rho_vap/R)*np.sqrt((R*T_if)/(2*math.pi))
    fac_dimless_r_21_extrapol = (rho_vap*T_if)*np.sqrt((R*T_if)/(2*math.pi))
    R_11_dimless = fac_dimless_r_11_extrapol*R_11
    R_21_dimless = fac_dimless_r_21_extrapol*R_21
    
    # Print solution
    if printSolution:
        yStr = ['T_if', 'rho_if', 'mu_if', 'T_vap', 'h_vap', 'mu_vap']
        for idx,value in enumerate(yStr):
            print(f'{value} = {sol[idx]}')
        print(f'R_11_anaSol = {R_11}')
        print(f'R_21_anaSol = {R_21}')
        print(f'R_11_anaSol_dimless = {R_11_dimless}')
        print(f'R_21_anaSol_dimless = {R_21_dimless}')
        
    return [rho_liq, h_liq, mu_liq, lambda_liq, rho_vap, T_if, rho_if, mu_if, T_vap, h_vap, mu_vap, R_11, R_21, R_11_dimless, R_21_dimless]


#%% Function to calculate interface resistivities based on kinetic theory by Chipolla 1976 and Xu, Kjelstrup et al. 2006
def kinTheory_ifRes(T, condCoeff=1.0, printSolution=True):
    '''
    Function to calculate interface resistivities based on kinetic theory

    :param float T: Liquid temperature / Interface temperature
    :param float condCoeff: Condensation coefficient
    :return: List including the calculated data
    '''
    
    # Direkte Berechungen
    _,rho_vap,_ = sat_vrabec2006(T)
    
    R = 1.0  # Gas constant
    M = 1.0  # Molar mass
    
    # Equation 11 from DOI: 10.1103/PhysRevE.75.061604
    R_11_v = ((4.34161*R*((1/condCoeff)-0.39856))/rho_vap)*np.sqrt(M/(3*R*T))  # r_uu
    R_12_v = R_21_v = (0.54715/(T*rho_vap))*np.sqrt(M/(3*R*T))  # r_uq and r_qu
    R_22_v = (1.27640/(R*(T**2)*rho_vap))*np.sqrt(M/(3*R*T))  # r_qq
    
    # Dimensionless according to Henning (Eq. 31 & 32)
    fac_dimless_r_11 =                    (rho_vap/R)*np.sqrt((R*T)/(2*math.pi))
    fac_dimless_r_12 = fac_dimless_r_21 = (rho_vap*T)*np.sqrt((R*T)/(2*math.pi))
    fac_dimless_r_22 =                    (rho_vap*R*(T**2))*np.sqrt((R*T)/(2*math.pi))
    R_11_v_dimless = fac_dimless_r_11*R_11_v
    R_12_v_dimless = fac_dimless_r_12*R_12_v
    R_21_v_dimless = fac_dimless_r_21*R_21_v
    R_22_v_dimless = fac_dimless_r_22*R_22_v
    
    # Print solution
    if printSolution:
        print(f'R_11_v_kinTheory = {R_11_v}')
        print(f'R_12_v_kinTheory = {R_12_v}')
        print(f'R_21_v_kinTheory = {R_21_v}')
        print(f'R_22_v_kinTheory = {R_22_v}')
        print(f'R_11_v_kinTheory_dimless = {R_11_v_dimless}')
        print(f'R_12_v_kinTheory_dimless = {R_12_v_dimless}')
        print(f'R_21_v_kinTheory_dimless = {R_21_v_dimless}')
        print(f'R_22_v_kinTheory_dimless = {R_22_v_dimless}')
    
    return [R_11_v, R_12_v, R_21_v, R_22_v, R_11_v_dimless, R_12_v_dimless, R_21_v_dimless, R_22_v_dimless]


#%% Function to calculate fluxes and other quantities analytically
def analSol_fluxes_evap(T_liq, v_z, L_n, R_11_v, R_12_v, R_21_v, R_22_v, printSolution=True):
    '''
    Function to calculate fluxes and other quantities analytically

    :param float T_liq: Liquid temperature
    :param float v_z: Hydrodynamic vapor velocity
    :param float L_n: Length of non-thermostated liquid
    :param float R_**: Resistivities
    :param bool printSolution: Print calculated values
    :return: List including the calculated data
    '''
    
    # Saturated liquid density based on Vrabec2006
    def rhol_vrabec2006(T):
        rc=0.3190
        dt=1.0779-T
        a,b,c=0.5649,0.1314,0.0413
        rhol=rc+a*dt**(1/3.)+b*dt+c*dt**(3/2.)
        return rhol
    
    # System of equations
    def equations(y):
        T_if, rho_if, mu_if, T_vap, rho_vap, h_vap, mu_vap, j_p, j_e = y
        eqs = [
            # equation 1
            rho_vap - j_p/v_z,
            # equation 2
            rho_if - rhol_vrabec2006(T_if),
            # equation 3
            mu_if - float(petspy.petseos(12,rho_if,19,T_if,51)),
            # equation 4
            h_vap - (float(petspy.petseos(12,rho_vap,19,T_vap,14))+2.0),  # Referenzpunkt
            # equation 5
            mu_vap - float(petspy.petseos(12,rho_vap,19,T_vap,51)),
            # equation 6
            j_e - ((h_vap + ((v_z**2)/2))*j_p),
            # equation 7
            j_e - (h_liq*j_p + lambda_liq*((T_liq-T_if)/L_n)),
            # equation 8
            j_p - ( O_uj*(((mu_if/T_if)-(mu_vap/T_vap)) + h_vap*((1/T_vap)-(1/T_if))) + O_Tj*((1/T_vap)-(1/T_if)) ),
            # equation 9
            0 - ( O_uq*(((mu_if/T_if)-(mu_vap/T_vap)) + h_vap*((1/T_vap)-(1/T_if))) + O_Tq*((1/T_vap)-(1/T_if)) ),
        ]
        return eqs
    
    # Direkte Berechungen
    rho_liq,_,_ = sat_vrabec2006(T_liq)
    h_liq = float(petspy.petseos(12,rho_liq,19,T_liq,14))+2.0  # Referenzpunkt
    mu_liq = float(petspy.petseos(12,rho_liq,19,T_liq,51))
    lambda_liq = lambda_lauten(T_liq,rho_liq)
    
    R_v = [[R_11_v, R_21_v], [R_12_v, R_22_v]]
    O_v = np.linalg.inv(R_v)
    
    if not np.allclose(np.dot(R_v, O_v), np.eye(2)): print('Warning! Inv. of R_v not closed')
    
    O_uj = O_v[0][0]
    O_Tj = O_v[0][1]
    O_uq = O_v[1][0]
    O_Tq = O_v[1][1]
    
    # Unbekannte und initial conditions: T_if, rho_if, mu_if, T_vap, rho_vap, h_vap, mu_vap, j_p, j_e
    T_if = 0.95*T_liq
    rho_if,rho_vap,_ = sat_vrabec2006(T_if)
    mu_if = float(petspy.petseos(12,rho_if,19,T_if,51))
    T_vap = 0.8*T_liq
    h_vap = float(petspy.petseos(12,rho_vap,19,T_vap,14))+2.0  # Referenzpunkt
    mu_vap = float(petspy.petseos(12,rho_vap,19,T_vap,51))
    j_p = rho_vap*v_z
    
    j_e = 1e-3   # First guess
    
    y0 = [T_if, rho_if, mu_if, T_vap, rho_vap, h_vap, mu_vap, j_p, j_e]
    
    # Solve the system of equations
    sol = fsolve(equations, y0)
    T_if, rho_if, mu_if, T_vap, rho_vap, h_vap, mu_vap, j_p, j_e = sol
    
    # Print solution
    if printSolution:
        yStr = ['T_if', 'rho_if', 'mu_if', 'T_vap', 'rho_vap', 'h_vap', 'mu_vap', 'j_p', 'j_e']
        for idx,value in enumerate(yStr):
            print(f'{value} = {sol[idx]}')
    
    return [rho_liq, h_liq, mu_liq, lambda_liq, T_if, rho_if, mu_if, T_vap, rho_vap, h_vap, mu_vap, j_p, j_e]


#%% Main program which can be called from the command line
if __name__ == '__main__':
    
    print('Do not call this file directly! Import functions instead.')
    