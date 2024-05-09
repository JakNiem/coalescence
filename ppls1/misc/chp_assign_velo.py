import ppls1.imp.chp as imp
import ppls1.exp.chp as exp
import numpy as np
import xml.etree.ElementTree as ET


#%% Function to reduce number of particles in checkpoint
def chp_assign_velo(in_file_path,out_file_path,in_array_temperature):
    '''
    Function to assign temperature to particles in a binary checkpoint using a profile array
    For the assignment, the checkpoint is divided in bins in y-direction

    :param str in_file_path:  Path and name of the binary checkpoint to be read
    :param str out_file_path: Path and name of the binary checkpoint to be written
    :param str in_array_temperature: Array data to be assigned to particles in chp
    '''
    
    print(f'Processing file: {in_file_path}')
    
    in_file_path_header = in_file_path[:-4]+'.header.xml'
    out_file_path_header = out_file_path[:-4]+'.header.xml'
    
    chp=imp.imp_chp_bin_DF(in_file_path)
    
    header = ET.parse(f'{in_file_path[:-4]}.header.xml')
    header_root = header.getroot()
    ymax = float(header_root.findall("./headerinfo/length/y")[0].text)
    
    for index, row in chp.iterrows():
        y = row['ry']
        vx = row['vx']
        vy = row['vy']
        vz = row['vz']
        binIdx = min(round((y/ymax)*len(in_array_temperature)), len(in_array_temperature)-1)
        T_set = in_array_temperature[binIdx]
        scale_fac = np.sqrt(3*T_set/(vx**2+vy**2+vz**2))
        chp.at[index, 'vx'] = scale_fac*vx
        chp.at[index, 'vy'] = scale_fac*vy
        chp.at[index, 'vz'] = scale_fac*vz
    
    
    # export checkpoint 
    exp.exp_chp_bin_DF(out_file_path,chp)
    
    headerXMLTree = ET.parse(in_file_path_header)
    headerXML = headerXMLTree.getroot()
    headerXML.find('headerinfo/time').text = str(0.0)
    headerXMLTree.write(out_file_path_header)
    
    print("   Values successfully assigned!")


#%% Main program which can be called from the command line
if __name__ == '__main__':
    
    flgInp=1 # 1: Parse args; 2: Hardcoded
    
    if flgInp==1:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('in_file_path', help='Path and name of the binary checkpoint to be read')
        parser.add_argument('out_file_path', help='Path and name of the binary checkpoint to be written')
        parser.add_argument('in_array_temperature', help='Array data to be assigned to particles in chp')
        args = parser.parse_args()
        in_file_path=args.in_file_path
        out_file_path=args.out_file_path
        in_array_temperature = np.fromstring(args.in_array_temperature, dtype=float)
    
    if flgInp==2:
        work_folder = '/home/pcfsuser/Simon/simulations/ls1/evaporation/1CLJTS/2022_Visualization/T084_v01_assignedValues/equi/'
        in_file_path = work_folder+'cp_binary-12.restart.dat'
        out_file_path = work_folder+'cp_binary-assign.restart.dat'
        in_array_temperature = np.linspace(0.7,2,20)
        
    
    chp_assign_velo(in_file_path,out_file_path,in_array_temperature)
