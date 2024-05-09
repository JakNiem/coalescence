import ppls1.imp.chp as imp
import ppls1.exp.chp as exp
import numpy as np
import xml.etree.ElementTree as ET


#%% Function to truncate box to given values
def chp_truncate_box(in_file_path,out_file_path, minBox, maxBox):
    '''
    Function to truncate box to given values using a binary checkpoint

    :param str in_file_path:  Path and name of the binary checkpoint to be read
    :param str out_file_path: Path and name of the binary checkpoint to be written
    '''
    
    print(f'Processing file: {in_file_path}')
    
    in_file_path_header = in_file_path[:-4]+'.header.xml'
    out_file_path_header = out_file_path[:-4]+'.header.xml'
    
    # Read in checkpoint
    chp = imp.imp_chp_bin_LD(in_file_path)
    
    headerXMLTree = ET.parse(in_file_path_header)
    headerXML = headerXMLTree.getroot()
    
    if maxBox[0] > float(headerXML.find('headerinfo/length/x').text):
        maxBox[0] = float(headerXML.find('headerinfo/length/x').text)
        print('Warning! Requested box length x too large. Setting to {maxBox[0]}.')
    if maxBox[1] > float(headerXML.find('headerinfo/length/y').text):
        maxBox[1] = float(headerXML.find('headerinfo/length/y').text)
        print('Warning! Requested box length y too large. Setting to {maxBox[1]}.')
    if maxBox[2] > float(headerXML.find('headerinfo/length/z').text):
        maxBox[2] = float(headerXML.find('headerinfo/length/z').text)
        print('Warning! Requested box length z too large. Setting to {maxBox[2]}.')
    
    partcomp = list()
    
    # check if inside new box
    for par in chp:
        if (par['rx'] <= maxBox[0]) and (par['rx'] >= minBox[0]):
            if (par['ry'] <= maxBox[1]) and (par['ry'] >= minBox[1]):
                if (par['rz'] <= maxBox[2]) and (par['rz'] >= minBox[2]):
                    par['rx'] = par['rx'] - minBox[0]
                    par['ry'] = par['ry'] - minBox[1]
                    par['rz'] = par['rz'] - minBox[2]
                    partcomp.append(par)
        
    # refresh particle ids
    for pi in range(len(partcomp)):
        partcomp[pi]['pid']=pi+1

    
    exp.exp_chp_bin_LD(out_file_path,partcomp)

    headerXML.find('headerinfo/length/x').text = str(maxBox[0]-minBox[0])
    headerXML.find('headerinfo/length/y').text = str(maxBox[1]-minBox[1])
    headerXML.find('headerinfo/length/z').text = str(maxBox[2]-minBox[2])
    headerXML.find('headerinfo/number').text = str(len(partcomp))
    headerXML.find('headerinfo/time').text = str(0.0)
    headerXMLTree.write(out_file_path_header)
    

#%% Main program which can be called from the command line
if __name__ == '__main__':
    
    flgInp=1 # 1: Parse args; 2: Hardcoded
    
    if flgInp==1:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('in_file_path', help='Path and name of the binary checkpoint to be read')
        parser.add_argument('out_file_path', help='Path and name of the binary checkpoint to be written')
        args = parser.parse_args()
        in_file_path=args.in_file_path
        out_file_path=args.out_file_path
    
    if flgInp==2:
        work_folder = '/home/pcfsuser/Simon/simulations/ls1/evaporation/1CLJTS/2022_Visualization/T084_v01/equi/'
        in_file_path = work_folder+'cp_binary-12.restart.dat'
        out_file_path = work_folder+'cp_binary-truncated.restart.dat'
        
    
    minBox = [ 0,   30,  0]  # x, y, z
    maxBox = [100, 400, 100]  # x, y, z
    
    chp_truncate_box(in_file_path, out_file_path, minBox, maxBox)
