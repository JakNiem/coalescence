import os
import math
from datetime import datetime
import random
import sys
import xml.etree.ElementTree as et
import ppls1.imp.chp as imp
import ppls1.exp.chp as exp
import numpy as np
import pandas as pd


#domain setup
r1 = 20 #radius of droplet 1
surfToBoundary = 10 #distance between surface and domain boundary
domainX_droplet = 2*r1 + 2*surfToBoundary

dList = 2 #distance between droplet surfaces

# physics setup
temperature = 0.7 

# simulation setup
vDroplet = [1] #for steps init & dropEqui, these are the droplet versions that are generated/worked on
vSelected = [1,1] # for production, these are the two selected droplets. if longer than two elements, the first two are always chosen
execStep = None
runls1 = True 

ls1_exec_relative = './ls1-mardyn/build/src/MarDyn'
ls1_exec = os.path.abspath(ls1_exec_relative)
# ls1_exec = '/home/niemann/ls1-mardyn_cylindricSampling/build/src/MarDyn'
stepName_bulk = "bulk" 
stepName_drop = "drop"
stepName_prod = "prod"
configName_bulk = "config_bulk.xml"
configName_drop = "config_drop.xml"
configName_prod = "config_prod.xml"


def main():
    global domainX_droplet 
    print(f'execstep: {execStep}')

    if(execStep == 'bulk'):
        domainX_droplet = 2*r1 + 2*surfToBoundary
        for v in vDroplet:
            work_dir = f"drp_T{temperature}_r{r1}_v{v}"
            print(f'workFolder: {work_dir}')
            step1_bulk(work_dir)  ## bulk liquid initialization & equi
            #TODO: bulk duplication for faster bulk equi

    elif execStep == 'drop':
        domainX_droplet = 2*r1 + 2*surfToBoundary
        for v in vDroplet:
            work_dir = f"drp_T{temperature}_r{r1}_v{v}"
            print(f'workFolder: {work_dir}')
            step2_drop(work_dir)  ## cutout & equi of single droplet

    elif execStep == 'prod':
        for d in dList:
            work_dir = f"T{temperature}_r{r1}_d{d}_v{vSelected[0]}.{vSelected[1]}"
            sourceDir1 = f"drp_T{temperature}_r{r1}_v{vSelected[0]}"
            sourceDir2 = f"drp_T{temperature}_r{r1}_v{vSelected[1]}"
            print(f'workFolder: {work_dir}')
            step3_prod(d, sourceDir1, sourceDir2, work_dir)  ## setup of coalescence and start production
    else:
        print(f'execStep argument "{execStep}" invalid. please specify correctly.')




def step1_bulk(work_dir = 'testDir'):
    rhol,rhov = vle_kedia2006(temperature)

    # # make sure sphereparams.xml exists
    # if not os.path.exists(os.path.normpath("./sphereparams.xml")):
    #     writeFile(template_sphereparams(), os.path.normpath("./sphereparams.xml"))    

    # create work_folder
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    # create conifg.xml

    bulkBox = domainX_droplet
    
    bulkConfigText = template_bulk(bulkBox, bulkBox, bulkBox, temperature, rhol)
    writeFile(bulkConfigText, os.path.join(work_dir, configName_bulk))

    # create bash (only stirling so far)
    bashName_bulkInit = 'stirling_bulk.sh'
    bashText = template_bash_drop(ls1_exec, configName_bulk, stepName_bulk, r1, temperature)
    writeFile(bashText, os.path.join(work_dir, bashName_bulkInit))
    os.system(f'chmod +x {os.path.join(work_dir, bashName_bulkInit)}')


    if runls1: os.system(f'cd {work_dir}; sbatch {bashName_bulkInit}')
    os.system('cd ..')

    return 0

def step2_drop(work_dir = 'testDir'):

    rhol,rhov = vle_kedia2006(temperature)
    
    in_file_path = os.path.join(work_dir, 'cp_binary_bulk-2.restart.dat')
    file_path_drop_start = os.path.join(work_dir, 'cp_binary_drop.start.dat')


    ############# adjust local densities
    in_file_path_header = in_file_path[:-4]+'.header.xml'
    file_path_drop_start_header = file_path_drop_start[:-4]+'.header.xml'
    
    # Read in checkpoint header
    headerXMLTree = et.parse(in_file_path_header)
    headerXML = headerXMLTree.getroot()
    
    # Read in checkpoint data
    chp = imp.imp_chp_bin_LD(in_file_path)
    
    # Get box lengths from xml header file
    xBox = float(headerXML.find('headerinfo/length/x').text)
    yBox = float(headerXML.find('headerinfo/length/y').text)
    zBox = float(headerXML.find('headerinfo/length/z').text)

    nParticles = float(headerXML.find('headerinfo/number').text)
    rhoBulk = nParticles/(xBox*yBox*zBox)
    if rhoBulk < rhol:
        print(f"WARNING! Bulk density too low, desired density of rhol={rhol} will not be achieved!")

    # Cutout droplets: 
    chpDroplet = []


    for par in chp:    # TODO: write faster method (using pd.sample)
        rx = par['rx'] 
        ry = par['ry']
        rz = par['rz']
        
        dx = rx - domainX_droplet/2
        dy = ry - domainX_droplet/2
        dz = rz - domainX_droplet/2
        dFromCenter = np.sqrt(dx**2 + dy**2 + dz**2)
        # Only keep some particles based on density profile
        if random.random() <= densityProfileSharp(dFromCenter, r1) / rhoBulk: 
            chpDroplet.append(par)
            
    num_new = len(chpDroplet)
    
    # refresh particle ids
    for pi in range(num_new):
        chpDroplet[pi]['pid']=pi+1
    
    headerXML.find('headerinfo/length/x').text = str(xBox)
    headerXML.find('headerinfo/length/y').text = str(yBox)
    headerXML.find('headerinfo/length/z').text = str(zBox)
    headerXML.find('headerinfo/number').text = str(num_new)
    headerXML.find('headerinfo/time').text = str(0.0)
    
    headerXMLTree.write(file_path_drop_start_header)
    exp.exp_chp_bin_LD(file_path_drop_start, chpDroplet)
    



    ############# start sim:
    # create conifg.xml
    dropConfigText = template_drop(xBox, yBox, zBox, temperature)
    writeFile(dropConfigText, os.path.join(work_dir, configName_drop))

    # create bash (only stirling so far):
    bashName_dropEqui = 'stirling_drop.sh'

    bashText = template_bash_drop(ls1_exec, configName_drop, stepName_drop, r1, temperature)

    writeFile(bashText, os.path.join(work_dir, bashName_dropEqui))
    os.system(f'chmod +x {os.path.join(work_dir, bashName_dropEqui)}')
    
    #run:
    if runls1: os.system(f'cd {work_dir}; sbatch {bashName_dropEqui}')





def step3_prod(d,sourceDir1, sourceDir2, work_dir = 'testDir'):
    
    # create work_folder
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)


    in_file_path_drop1 = os.path.join(sourceDir1, 'cp_binary_drop-2.restart.dat')
    in_file_path_drop2 = os.path.join(sourceDir2, 'cp_binary_drop-2.restart.dat')
    file_path_prod_start = os.path.join(work_dir, 'cp_binary_prod.start.dat')


    ############# adjust local densities
    in_file_path_header_drop1 = in_file_path_drop1[:-4]+'.header.xml'
    # in_file_path_header_drop2 = in_file_path_drop2[:-4]+'.header.xml'
    file_path_prod_start_header = file_path_prod_start[:-4]+'.header.xml'
    
    # Read in checkpoint header
    headerXMLTree = et.parse(in_file_path_header_drop1)
    headerXML = headerXMLTree.getroot()
    
    
    # Get box lengths from xml header file
    xBoxDrop = float(headerXML.find('headerinfo/length/x').text)
    yBoxDrop = float(headerXML.find('headerinfo/length/y').text)
    zBoxDrop = float(headerXML.find('headerinfo/length/z').text)
    if yBoxDrop != xBoxDrop:
        print(f"WARNING: step3: yboxDrop != xBoxDrop.")
    if zBoxDrop != xBoxDrop:
        print(f"WARNING: step3: zboxDrop != xBoxDrop.")

    finClearence = .5 # minimum distance of particles from stiching-plane 
    if finClearence>=d/2:
        print(f"WARNING! reducing finClearence to d/2 =  {d/2} to adapt to small droplet distance of d = {d}.")
        finClearence = d/2
    
    centerDomainDrop = xBoxDrop/2
    yBoxSubdomain = centerDomainDrop+r1+d/2
    yBoxFull = yBoxSubdomain *2

    # Read in checkpoint data to create two droplet domains
    #df Head:    pid  cid	rx	ry	rz	vx	vy	vz	q0	q1	q2	q3	Dx	Dy	Dz
    dfDrop1 = imp.imp_chp_bin_DF(in_file_path_drop1) 
    dfDrop2 = imp.imp_chp_bin_DF(in_file_path_drop2) 

    dfDrop1_reduced = dfDrop1[dfDrop1['ry'] <= (yBoxSubdomain - finClearence)]
    dfDrop2_reduced = dfDrop2[dfDrop2['ry'] >= (yBoxDrop-yBoxSubdomain+ finClearence)]
    dfDrop2_reduced['ry'] = dfDrop2_reduced['ry'] + (yBoxSubdomain - (yBoxDrop-yBoxSubdomain))
    
    dfFull = pd.concat([dfDrop1_reduced, dfDrop2_reduced])
    # print(5)
    # print(dfFull)

    # refresh index and particle ids 
    num_new = len(dfFull)
    idArr = np.array(range(len(dfFull)))
    pidArr = idArr +1
    dfFull['pid'] = pidArr

    

    # ## TODO: for testing: plot once: 
    # import matplotlib.pyplot as plt
    # ax1 = dfFull.plot.scatter('rx', 'ry')
    # ax1.set_aspect('equal', 'box')
    # plt.savefig('fig1.png')

    dfFull = dfFull.set_index(idArr)
    # print(dfFull.to_string())

    headerXML.find('headerinfo/length/x').text = str(xBoxDrop)
    headerXML.find('headerinfo/length/y').text = str(yBoxFull)
    headerXML.find('headerinfo/length/z').text = str(zBoxDrop)
    headerXML.find('headerinfo/number').text = str(num_new)
    headerXML.find('headerinfo/time').text = str(0.0)
    
    headerXMLTree.write(file_path_prod_start_header)
    exp.exp_chp_bin_DF(file_path_prod_start, dfFull)

    



    # create conifg.xml
    prodConfigText = template_prod(xBoxDrop, yBoxFull, zBoxDrop, temperature)
    writeFile(prodConfigText, os.path.join(work_dir, configName_prod))

    # create bash (only stirling so far):
    bashName_prod = 'stirling_prod.sh'
    bashText = template_bash(ls1_exec, configName_prod, stepName_prod, d, temperature)

    writeFile(bashText, os.path.join(work_dir, bashName_prod))
    os.system(f'chmod +x {os.path.join(work_dir, bashName_prod)}')

    #run:
    if runls1: os.system(f'cd {work_dir}; sbatch {bashName_prod}')








#################### FUNCTION TOOLBOX ####################


def vle_kedia2006(T):
    '''
    Get saturated densities of DROPLET by Vrabec et al., Molecular Physics 104 (2006). Equation numbers refer this paper.
    :param float T: Temperature
    :return: float rhol, float rhov: Saturated liquid and vapor density
    '''
    Tc = 1.0779
    rc = 0.3190

    dT = (Tc-T)

    a,b,c=0.5649,0.1314,0.0413
    rhol=rc+a*dT**(1/3.)+b*dT+c*dT**(3/2.)       # equation 4
    a,b,c=0.5649,0.2128,0.0702
    rhov=rc-a*dT**(1/3.)+b*dT+c*dT**(3/2.)       # equation 5

    return rhol,rhov




def densityProfileSharp(dCenter, r):
    rhol,rhov = vle_kedia2006(temperature)
    # dCenter := distance from center of droplet 1
    # naive density profile.
    if dCenter <= r1:
        return rhol
    else:
        return rhov
    





def writeFile(content, outfilepath):
    f = open(outfilepath, 'w')
    f.write(content)
    f.close


#################### TEMPLATES ####################
def template_bash_drop(ls1Exec, configName, stepName, r, temperature, nodes = 1, nTasks = 1, ntasksPerNode = 1, cpusPerTask = 1):
    return f"""#!/bin/sh

#SBATCH -J {stepName}_r{r}_T{temperature}
#SBATCH --nodes={nodes}

### 1*8 MPI ranks
#SBATCH --ntasks={nTasks}

### 128/16 MPI ranks per node
#SBATCH --ntasks-per-node={ntasksPerNode}

### tasks per MPI rank
#SBATCH --cpus-per-task={cpusPerTask}

#SBATCH -e ./err_{stepName}.%j.log
#SBATCH -o ./out_{stepName}.%j.log

{ls1Exec} --final-checkpoint=1 {configName} 
    """       

def template_bash(ls1Exec, configName, stepName, d, temperature, nodes = 1, nTasks = 1, ntasksPerNode = 1, cpusPerTask = 1):
    return f"""#!/bin/sh

#SBATCH -J coa_d{d}_T{temperature}
#SBATCH --nodes={nodes}

### 1*8 MPI ranks
#SBATCH --ntasks={nTasks}

### 128/16 MPI ranks per node
#SBATCH --ntasks-per-node={ntasksPerNode}

### tasks per MPI rank
#SBATCH --cpus-per-task={cpusPerTask}

#SBATCH -e ./err_{stepName}.%j.log
#SBATCH -o ./out_{stepName}.%j.log

{ls1Exec} --final-checkpoint=1 {configName} 
    """       




# def template_sphereparams():
#     return f"""<?xml version='1.0' encoding='UTF-8'?>
# <spheres>
#     <!-- 1CLJTS -->
#     <site id="1">
#         <radius>0.5</radius>
#         <color>
#             <r>0</r>
#             <g>0</g>
#             <b>155</b>
#             <alpha>255</alpha>
#         </color>
#     </site>
# </spheres>
#     """        
	

	
def template_bulk(boxx, boxy, boxz, temperature, rhol):
    """
    returns the contents of the config.xml file for step1_bulk:
    -- bulk liquid with densitiy rho = rhol*1.1
    -- 1000 steps for equilibriation
    """

    simsteps = int(1000)
    writefreq = int(simsteps/2)
    density  = rhol*1.1
    return f"""<?xml version='1.0' encoding='UTF-8'?>
<mardyn version="20100525" >

<refunits type="SI">
    <length unit="nm">0.1</length>
    <mass unit="u">1</mass>
    <energy unit="K">1</energy>
</refunits>

<simulation type="MD" >            
    <integrator type="Leapfrog" >
        <timestep unit="reduced" >0.004</timestep>
    </integrator>

    <run>
        <currenttime>0</currenttime>
        <production>
            <steps>{simsteps}</steps>
        </production>
    </run>

    <ensemble type="NVT">
        <temperature unit="reduced" >{temperature}</temperature>
        <domain type="box">
            <lx>{boxx}</lx>
            <ly>{boxy}</ly>
            <lz>{boxz}</lz>
        </domain>

        <components>
            <!-- 1CLJTS -->
            <moleculetype id="1" name="1CLJTS">
                <site type="LJ126" id="1" name="LJTS">
                    <coords> <x>0.0</x> <y>0.0</y> <z>0.0</z> </coords>
                    <mass>1.0</mass>
                    <sigma>1.0</sigma>
                    <epsilon>1.0</epsilon>
                    <shifted>true</shifted>
                </site>
                <momentsofinertia rotaxes="xyz" >
                    <Ixx>0.0</Ixx>
                    <Iyy>0.0</Iyy>
                    <Izz>0.0</Izz>
                </momentsofinertia>
            </moleculetype>
        </components>
    
        <phasespacepoint>
            <generator name="CubicGridGenerator">
                <specification>density</specification>
                <density>{density}</density>
                <binaryMixture>false</binaryMixture>
            </generator>
        </phasespacepoint>

    </ensemble>

    <algorithm>
        <parallelisation type="DomainDecomposition">
            <!--<MPIGridDims> <x>2</x> <y>2</y> <z>2</z> </MPIGridDims>-->
        </parallelisation>
        <datastructure type="LinkedCells">
            <cellsInCutoffRadius>1</cellsInCutoffRadius>
        </datastructure>
        <cutoffs type="CenterOfMass" >
            <defaultCutoff unit="reduced" >2.5</defaultCutoff>
            <radiusLJ unit="reduced" >2.5</radiusLJ>
        </cutoffs>
        <electrostatic type="ReactionField" >
            <epsilon>1.0e+10</epsilon>
        </electrostatic>

        <longrange type="none">
        </longrange>


        <thermostats>
            <thermostat type="TemperatureControl">
                <control>
                    <start>0</start>
                    <frequency>1</frequency>
                    <stop>1000000000</stop>
                </control>
                <regions>
                    <region>
                        <coords>
                            <lcx>0.0</lcx> <lcy>0.0</lcy> <lcz>0.0</lcz>
                            <ucx>box</ucx> <ucy>box</ucy> <ucz>box</ucz>
                        </coords>
                        <target>
                            <temperature>{temperature}</temperature>
                            <component>0</component>
                        </target>
                        <settings>
                            <numslabs>1</numslabs>
                            <exponent>0.4</exponent>
                            <directions>xyz</directions>
                        </settings>
                        <writefreq>100</writefreq>
                        <fileprefix>temp_log</fileprefix>
                    </region>
                </regions>
            </thermostat>
        </thermostats> 
    </algorithm>

    <output>
        <outputplugin name="CheckpointWriter">
            <type>binary</type>
            <writefrequency>{writefreq}</writefrequency>
            <outputprefix>cp_binary_bulk</outputprefix>
        </outputplugin>
    </output>

    <plugin name="COMaligner">
		<x>true</x>
		<y>true</y>
		<z>true</z>
		<interval>10</interval>
		<correctionFactor>1.0</correctionFactor>
	</plugin>

    <plugin name="DriftCtrl">
        <control>
            <start>0</start>
            <stop>20000000</stop>
            <freq>
                <sample>100</sample>
                <control>100</control>
                <write>100</write>
            </freq>
        </control>
        <target>
            <cid>1</cid>
            <drift> <vx>0.0</vx> <vy>0.0</vy> <vz>0.0</vz> </drift>
        </target>
        <range>
            <yl>0</yl> <yr>24</yr>
            <subdivision>
                <binwidth>24</binwidth>
            </subdivision>
        </range>
    </plugin>

</simulation>
</mardyn>
"""


def template_drop(boxx, boxy, boxz, temperature):
    simsteps = int(10e3)
    writefreq = int(5e3)
    # mmpldFreq = int(500)
    # rsfreq = int(mmpldFreq)
    # cylSamplingFreq = int(500)
    return f"""<?xml version='1.0' encoding='UTF-8'?>
<mardyn version="20100525" >

<refunits type="SI">
    <length unit="nm">0.1</length>
    <mass unit="u">1</mass>
    <energy unit="K">1</energy>
</refunits>

<simulation type="MD" >            
    <integrator type="Leapfrog" >
        <timestep unit="reduced" >0.004</timestep>
    </integrator>

    <run>
        <currenttime>0</currenttime>
        <production>
            <steps>{simsteps}</steps>
        </production>
    </run>

    <ensemble type="NVT">
        <temperature unit="reduced" >{temperature}</temperature>
        <domain type="box">
            <lx>{boxx}</lx>
            <ly>{boxy}</ly>
            <lz>{boxz}</lz>
        </domain>

        <components>
            <!-- 1CLJTS -->
            <moleculetype id="1" name="1CLJTS">
                <site type="LJ126" id="1" name="LJTS">
                    <coords> <x>0.0</x> <y>0.0</y> <z>0.0</z> </coords>
                    <mass>1.0</mass>
                    <sigma>1.0</sigma>
                    <epsilon>1.0</epsilon>
                    <shifted>true</shifted>
                </site>
                <momentsofinertia rotaxes="xyz" >
                    <Ixx>0.0</Ixx>
                    <Iyy>0.0</Iyy>
                    <Izz>0.0</Izz>
                </momentsofinertia>
            </moleculetype>
        </components>
    
        <phasespacepoint>
			<file type="binary">
				<header>./cp_binary_drop.start.header.xml</header>
				<data>./cp_binary_drop.start.dat</data>
			</file>
			<ignoreCheckpointTime>true</ignoreCheckpointTime>
        </phasespacepoint>

    </ensemble>

    <algorithm>
        <parallelisation type="DomainDecomposition">
            <!--<MPIGridDims> <x>2</x> <y>2</y> <z>2</z> </MPIGridDims>-->
        </parallelisation>
        <datastructure type="LinkedCells">
            <cellsInCutoffRadius>1</cellsInCutoffRadius>
        </datastructure>
        <cutoffs type="CenterOfMass" >
            <defaultCutoff unit="reduced" >2.5</defaultCutoff>
            <radiusLJ unit="reduced" >2.5</radiusLJ>
        </cutoffs>
        <electrostatic type="ReactionField" >
            <epsilon>1.0e+10</epsilon>
        </electrostatic>

        <longrange type="none">
        </longrange>


        <thermostats>
            <thermostat type="TemperatureControl">
                <control>
                    <start>0</start>
                    <frequency>1</frequency>
                    <stop>1000000000</stop>
                </control>
                <regions>
                    <region>
                        <coords>
                            <lcx>0.0</lcx> <lcy>0.0</lcy> <lcz>0.0</lcz>
                            <ucx>box</ucx> <ucy>box</ucy> <ucz>box</ucz>
                        </coords>
                        <target>
                            <temperature>{temperature}</temperature>
                            <component>0</component>
                        </target>
                        <settings>
                            <numslabs>1</numslabs>
                            <exponent>0.4</exponent>
                            <directions>xyz</directions>
                        </settings>
                        <writefreq>10000</writefreq>
                        <fileprefix>temp_log</fileprefix>
                    </region>
                </regions>
            </thermostat>
        </thermostats> 
    </algorithm>

    <output>
        <outputplugin name="CheckpointWriter">
            <type>binary</type>
            <writefrequency>{writefreq}</writefrequency>
            <outputprefix>cp_binary_drop</outputprefix>
        </outputplugin>
    </output>

    <plugin name="COMaligner">
		<x>true</x>
		<y>true</y>
		<z>true</z>
		<interval>10</interval>
		<correctionFactor>1.0</correctionFactor>
	</plugin>
    
    <plugin name="DriftCtrl">
        <control>
            <start>0</start>
            <stop>20000000</stop>
            <freq>
                <sample>100</sample>
                <control>100</control>
                <write>100</write>
            </freq>
        </control>
        <target>
            <cid>1</cid>
            <drift> <vx>0.0</vx> <vy>0.0</vy> <vz>0.0</vz> </drift>
        </target>
        <range>
            <yl>0</yl> <yr>24</yr>
            <subdivision>
                <binwidth>24</binwidth>
            </subdivision>
        </range>
    </plugin>
</simulation>
</mardyn>
"""



def template_prod(boxx, boxy, boxz, temperature):
    simsteps = int(30e3)
    writefreq = int(5e3)
    mmpldFreq = int(500)
    rsfreq = int(mmpldFreq)
    cylSamplingFreq = int(400)   
    #coalSampling:
    coalSamplingFreq = int(100)
    coalSamplingBindwidth = .25
    coalSamplingRadius = 4
    coalOutputPrefix = 'coalSampling_'
    #cuboidSampling:
    cuboidSampingWidth = 2.5
    numbinsCubSamp = 100

    return f"""<?xml version='1.0' encoding='UTF-8'?>
<mardyn version="20100525" >

<refunits type="SI">
    <length unit="nm">0.1</length>
    <mass unit="u">1</mass>
    <energy unit="K">1</energy>
</refunits>

<simulation type="MD" >            
    <integrator type="Leapfrog" >
        <timestep unit="reduced" >0.004</timestep>
    </integrator>

    <run>
        <currenttime>0</currenttime>
        <production>
            <steps>{simsteps}</steps>
        </production>
    </run>

    <ensemble type="NVT">
        <temperature unit="reduced" >{temperature}</temperature>
        <domain type="box">
            <lx>{boxx}</lx>
            <ly>{boxy}</ly>
            <lz>{boxz}</lz>
        </domain>

        <components>
            <!-- 1CLJTS -->
            <moleculetype id="1" name="1CLJTS">
                <site type="LJ126" id="1" name="LJTS">
                    <coords> <x>0.0</x> <y>0.0</y> <z>0.0</z> </coords>
                    <mass>1.0</mass>
                    <sigma>1.0</sigma>
                    <epsilon>1.0</epsilon>
                    <shifted>true</shifted>
                </site>
                <momentsofinertia rotaxes="xyz" >
                    <Ixx>0.0</Ixx>
                    <Iyy>0.0</Iyy>
                    <Izz>0.0</Izz>
                </momentsofinertia>
            </moleculetype>
        </components>
    
        <phasespacepoint>
			<file type="binary">
				<header>./cp_binary_prod.start.header.xml</header>
				<data>./cp_binary_prod.start.dat</data>
			</file>
			<ignoreCheckpointTime>true</ignoreCheckpointTime>
        </phasespacepoint>

    </ensemble>

    <algorithm>
        <parallelisation type="DomainDecomposition">
            <!--<MPIGridDims> <x>2</x> <y>2</y> <z>2</z> </MPIGridDims>-->
        </parallelisation>
        <datastructure type="LinkedCells">
            <cellsInCutoffRadius>1</cellsInCutoffRadius>
        </datastructure>
        <cutoffs type="CenterOfMass" >
            <defaultCutoff unit="reduced" >2.5</defaultCutoff>
            <radiusLJ unit="reduced" >2.5</radiusLJ>
        </cutoffs>
        <electrostatic type="ReactionField" >
            <epsilon>1.0e+10</epsilon>
        </electrostatic>

        <longrange type="none">
        </longrange>


        <thermostats>
        </thermostats> 
    </algorithm>

    <output>
        <outputplugin name="CheckpointWriter">
            <type>binary</type>
            <writefrequency>{writefreq}</writefrequency>
            <outputprefix>cp_binary_prod</outputprefix>
        </outputplugin>
        <outputplugin name="MmpldWriter" type="multi">
            <spheres>
                <!-- 1CLJTS -->
                <site id="1">
                    <radius>0.5</radius>
                    <color>
                        <r>0</r>
                        <g>0</g>
                        <b>155</b>
                        <alpha>255</alpha>
                    </color>
                </site>
            </spheres>
			<writecontrol>
				<start>0</start>
				<writefrequency>{mmpldFreq}</writefrequency>
				<stop>500000000</stop>
				<framesperfile>0</framesperfile>
			</writecontrol>
			<outputprefix>megamol</outputprefix>
		</outputplugin>
		<outputplugin name="Adios2Writer">
			<outputfile>adios_checkpoint.bp</outputfile>
			<adios2enginetype>BP4</adios2enginetype>
			<writefrequency>{mmpldFreq}</writefrequency>
		</outputplugin>
    </output>
    

    <plugin name="COMaligner">
		<x>true</x>
		<y>true</y>
		<z>true</z>
		<interval>10</interval>
		<correctionFactor>1.0</correctionFactor>
	</plugin>
    
    <plugin name="DriftCtrl">
        <control>
            <start>0</start>
            <stop>20000000</stop>
            <freq>
                <sample>100</sample>
                <control>100</control>
                <write>100</write>
            </freq>
        </control>
        <target>
            <cid>1</cid>
            <drift> <vx>0.0</vx> <vy>0.0</vy> <vz>0.0</vz> </drift>
        </target>
        <range>
            <yl>0</yl> <yr>24</yr>
            <subdivision>
                <binwidth>24</binwidth>
            </subdivision>
        </range>
    </plugin>

    <plugin name="CylindricSampling">
        <binwidth>.5</binwidth>                  <!-- Width of sampling bins; default 1.0 -->
        <start>0</start>                          <!-- Simstep to start sampling; default 0 -->
        <writefrequency>{cylSamplingFreq}</writefrequency>        <!-- Simstep to write out result file; default 10000 -->
        <stop>1000000000000</stop>                            <!-- Simstep to stop sampling; default 1000000000 -->
    </plugin>

    <plugin name="CoalescenceSampling">
        <binwidth>{coalSamplingBindwidth}</binwidth>                                   <!-- Width of sampling bins; default 1.0 -->
        <radius>{coalSamplingRadius}</radius>                      <!-- radius of Cylinder -->
        <start>0</start>                                         <!-- Simstep to start sampling; default 0 -->
        <writefrequency>{coalSamplingFreq}</writefrequency>        <!-- Simstep to write out result file; default 100 -->
        <yLower>0</yLower>                                        <!-- start of Cylinder -->
        <yUpper>{boxy}</yUpper>                                       <!-- end of Cylinder -->
        <outputPrefix>{coalOutputPrefix}</outputPrefix>                      <!-- outputPrefix -->
    </plugin>

    <plugin name="CuboidSampling">
        <numBinsX>{numbinsCubSamp}</numBinsX>                  <!-- # of sampling bins; default 100 -->
        <numBinsZ>{numbinsCubSamp}</numBinsZ>                  <!-- # of sampling bins; default 100 -->
        <writefrequency>{coalSamplingFreq}</writefrequency>        <!-- Simstep to write out result file; default 100 -->
        <yLower>{boxy/2 - cuboidSampingWidth/2}</yLower>                      <!-- start y-Direction -->
        <yUpper>{boxy/2 + cuboidSampingWidth/2}</yUpper>                      <!-- stop y-direction -->
    </plugin>
</simulation>
</mardyn>
"""

#################### END OF TEMPLATES ####################










#################### CALLING MAIN ####################
# Call main() at the bottom, after all the functions have been defined.
if __name__ == '__main__':

    # # ARGUMENTS: give arguments like the following:
    # # m1: => use procedure for machine 1 (stirling)
    # # "init": => run step "init"
    # # T1 or T.8,.9,1: => TList = ...
    # # r8 or r8,9,10: => rList = ...


    ### Reading Arguments
    for arg in sys.argv[1:]:    #argument #0 is always [scriptname].py
        if arg in ['init',  'bulk']:
            execStep = 'bulk'
        elif arg in ['equi', 'drop']:
            execStep = 'drop'
        elif arg in ['prod', 'coal']:
            execStep = 'prod'
        elif arg == 'noRun':
            runls1 = False
        elif arg.startswith('T'):  
            if len(arg) < 3 or type(eval(arg[1:])) == type(1) or type(eval(arg[1:]))== type(.7): arg+=',' 
            temperature = list(eval(arg[1:]))[0]
            print(f'argument {arg} interpreted as T = {temperature}')
        elif arg.startswith('d'):
            if len(arg) < 3 or type(eval(arg[1:])) == type(1) or type(eval(arg[1:]))== type(.7): arg+=',' 
            dList = list(eval(arg[1:]))
            print(f'argument {arg} interpreted as d = {dList}')
        elif arg.startswith('r'):
            if len(arg) < 3 or type(eval(arg[1:])) == type(1) or type(eval(arg[1:]))== type(.7): arg+=',' 
            r1 = list(eval(arg[1:]))[0]
            print(f'argument {arg} interpreted as r1 = {r1}')
        # elif arg.startswith('vuse'):  
        #     if len(arg) < 3 or type(eval(arg[1:])) == type(1) or type(eval(arg[1:]))== type(.7): arg+=',' 
        #     vSelected = list(eval(arg[4:]))
        #     print(f'argument {arg} interpreted as vSelected = {vSelected}')
        elif arg.startswith('v'):  
            if len(arg) < 3 or type(eval(arg[1:])) == type(1) or type(eval(arg[1:]))== type(.7): arg+=',' 
            vDroplet = list(eval(arg[1:]))
            vSelected = list(eval(arg[1:]))
            print(f'argument {arg} interpreted as vDroplet = {vDroplet}. argument only used if step is init or equi.')
            print(f'argument {arg} interpreted as vSelected = {vSelected}')
        # elif arg.startswith('n'):
        #     if len(arg) < 3 or type(eval(arg[1:])) == type(1) or type(eval(arg[1:]))== type(.7): arg+=',' 
        #     rerunNumber = list(eval(arg[1:]))[0]
        #     print(f'argument {arg} interpreted as rerunNumber = {rerunNumber}')
    
    
    #     elif arg == 'test':
    #         work_folder = os.path.join(work_folder, f'test')  
    #     elif arg.startswith('m'):
    #         machine = int(eval(arg[1:]))
    #         if machine in [0,1,2]:
    #             print(f'argument {arg} interpreted as machine = {machine}')
    #         else:
    #             print(f'invalid machine argument [{machine}]. setting machine = 3')
    #             machine = 3
        else:
            print(f'arg {arg} not a valid argument')

    main()
