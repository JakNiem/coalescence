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


#TODO: check domain decomposition --> 2x2x2 sollte auch für production (coalesence) optimal sein, wenn y-symmetrisch

#domain setup
r1 = 6 #radius of droplet 1

d = 4 #distance between droplet surfaces
surfToBoundary = 3*r1 #distance between surface and domain boundary
domainX = 2*r1 + 2*surfToBoundary
# domainZ = domainX
# domainY = (2*r1 + surfToBoundary)*2 + d
# #derived positional values:
# center1Y = surfToBoundary + r1
# center2Y = domainY - surfToBoundary - r1

# physics setup
temperature = .7 



# simulation setup
execStep = None

ls1_exec = '/home/niemann/ls1-mardyn_cylindricSampling/build/src/MarDyn'
work_folder = f"T{temperature}_d{d}_r{r1}" #_{str(datetime.now()).replace(' ', '')}
stepName_init = "init" #TODO: rename steps: bulk -- drop -- prod/coal
stepName_equi = "equi"
stepName_prod = "prod"
configName_init = "config_init.xml"
configName_equi = "config_equi.xml"
configName_prod = "config_prod.xml"

def main():
    if(execStep == 'bulk'):
        step1_init()  ## bulk liquid initialization & equi
        #TODO: bulk duplication for faster bulk equi
    elif execStep == 'drop':
        step2_equi()  ## cutout & equi of single droplet
    elif execStep == 'prod':
        step3_prod()  ## setup of coalescence and start production
    else:
        print(f'execStep argument "{execStep}" invalid. please specify correctly.')

def step1_init():
    rhol,rhov = vle_kedia2006(temperature)

    # # make sure sphereparams.xml exists
    # if not os.path.exists(os.path.normpath("./sphereparams.xml")):
    #     writeFile(template_sphereparams(), os.path.normpath("./sphereparams.xml"))    

    # create work_folder
    if not os.path.exists(work_folder):
        os.mkdir(work_folder)
    # create conifg.xml

    bulkBox = domainX
    
    bulkConfigText = template_init(bulkBox, bulkBox, bulkBox, temperature, rhol)
    writeFile(bulkConfigText, os.path.join(work_folder, configName_init))

    # create bash (only stirling so far)
    bashName_bulkInit = 'stirling_bulk.sh'
    bashText = template_bash(ls1_exec, configName_init, stepName_init)
    writeFile(bashText, os.path.join(work_folder, bashName_bulkInit))
    os.system(f'chmod +x {os.path.join(work_folder, bashName_bulkInit)}')


    os.system(f'cd {work_folder}; sbatch {bashName_bulkInit}')
    os.system('cd ..')

    return 0

def step2_equi():

    rhol,rhov = vle_kedia2006(temperature)
    
    in_file_path = os.path.join(work_folder, 'cp_binary_bulk-2.restart.dat')
    file_path_equi_start = os.path.join(work_folder, 'cp_binary_equi.start.dat')


    ############# adjust local densities
    in_file_path_header = in_file_path[:-4]+'.header.xml'
    file_path_equi_start_header = file_path_equi_start[:-4]+'.header.xml'
    
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
        
        dx = rx - domainX/2
        dy = ry - domainX/2
        dz = rz - domainX/2
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
    
    headerXMLTree.write(file_path_equi_start_header)
    exp.exp_chp_bin_LD(file_path_equi_start, chpDroplet)
    



    ############# start sim:
    # create conifg.xml
    equiConfigText = template_equi(xBox, yBox, zBox, temperature)
    writeFile(equiConfigText, os.path.join(work_folder, configName_equi))

    # create bash (only stirling so far):
    bashName_dropEqui = 'stirling_drop.sh'
    bashText = template_bash(ls1_exec, configName_equi, stepName_equi)

    writeFile(bashText, os.path.join(work_folder, bashName_dropEqui))
    os.system(f'chmod +x {os.path.join(work_folder, bashName_dropEqui)}')
    
    #run:
    os.system(f'cd {work_folder}; sbatch {bashName_dropEqui}')





def step3_prod():


    in_file_path = os.path.join(work_folder, 'cp_binary_drop-2.restart.dat')
    file_path_prod_start = os.path.join(work_folder, 'cp_binary_prod.start.dat')


    ############# adjust local densities
    in_file_path_header = in_file_path[:-4]+'.header.xml'
    file_path_prod_start_header = file_path_prod_start[:-4]+'.header.xml'
    
    # Read in checkpoint header
    headerXMLTree = et.parse(in_file_path_header)
    headerXML = headerXMLTree.getroot()
    
    
    # Get box lengths from xml header file
    xBoxDrop = float(headerXML.find('headerinfo/length/x').text)
    yBoxDrop = float(headerXML.find('headerinfo/length/y').text)
    zBoxDrop = float(headerXML.find('headerinfo/length/z').text)
    if yBoxDrop != xBoxDrop:
        print(f"WARNING: step3: yboxDrop != xBoxDrop.")
    if zBoxDrop != xBoxDrop:
        print(f"WARNING: step3: zboxDrop != xBoxDrop.")

    vacuumslab = .5 # minimum distance of particles from stiching-plane 
    if vacuumslab>=d/2:
        print(f"ERROR! vacuumslab>=d. exiting.")
        exit()
    
    centerDomainDrop = xBoxDrop/2
    yBoxSubdomain = centerDomainDrop+r1+d/2
    yBoxFull = yBoxSubdomain *2

    # Read in checkpoint data to create two droplet domains
    dfDrop1 = imp.imp_chp_bin_DF(in_file_path)  #pid  cid	rx	ry	rz	vx	vy	vz	q0	q1	q2	q3	Dx	Dy	Dz
    print(dfDrop1)
    dfDrop2 = dfDrop1.copy(deep=True)
    print(1)
    print(dfDrop2)

    dfDrop1_reduced = dfDrop1[dfDrop1['ry'] <= (yBoxSubdomain - vacuumslab)]
    print(2)
    print(dfDrop1_reduced)
    dfDrop2_reduced = dfDrop1[dfDrop2['ry'] >= (yBoxDrop-yBoxSubdomain+ vacuumslab)]
    print(3)
    print(dfDrop2_reduced)
    dfDrop2_reduced['ry'] = dfDrop2_reduced['ry'] + (yBoxSubdomain - (yBoxDrop-yBoxSubdomain))
    print(4)
    print(dfDrop2_reduced)
    
    dfFull = pd.concat([dfDrop1_reduced, dfDrop2_reduced])
    print(5)
    print(dfFull)


    ## for testing: plot once: 



    num_new = len(dfFull)
    
    # refresh particle ids
    for pi in range(num_new):
        dfFull['pid'][pi]=pi+1
    print(6)
    print(dfFull)
    print(dfFull.head())

    headerXML.find('headerinfo/length/x').text = str(xBoxDrop)
    headerXML.find('headerinfo/length/y').text = str(yBoxFull)
    headerXML.find('headerinfo/length/z').text = str(zBoxDrop)
    headerXML.find('headerinfo/number').text = str(num_new)
    headerXML.find('headerinfo/time').text = str(0.0)
    
    headerXMLTree.write(file_path_prod_start_header)
    exp.exp_chp_bin_DF(file_path_prod_start, dfFull)  ##TODO: this produces error. check out why.

    







    # create conifg.xml
    prodConfigText = template_prod(xBox, yBox, zBox, temperature)
    writeFile(prodConfigText, os.path.join(work_folder, configName_prod))

    # create bash (only stirling so far):
    bashName_prod = 'stirling_prod.sh'
    bashText = template_bash(ls1_exec, configName_prod, stepName_prod)

    writeFile(bashText, os.path.join(work_folder, bashName_prod))
    os.system(f'chmod +x {os.path.join(work_folder, bashName_prod)}')

    #run:
    os.system(f'cd {work_folder}; sbatch {bashName_prod}')








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
def template_bash(ls1Exec, configName, stepName, nodes = 1, nTasks = 1, ntasksPerNode = 1, cpusPerTask = 1):
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
	

	
def template_init(boxx, boxy, boxz, temperature, rhol):
    """
    returns the contents of the config.xml file for step1_init:
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


def template_equi(boxx, boxy, boxz, temperature):
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
				<header>./cp_binary_equi.start.header.xml</header>
				<data>./cp_binary_equi.start.dat</data>
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
    simsteps = int(10e3)
    writefreq = int(5e3)
    mmpldFreq = int(500)
    rsfreq = int(mmpldFreq)
    cylSamplingFreq = int(500)
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
            <binwidth>2</binwidth>                  <!-- Width of sampling bins; default 1.0 -->
            <start>0</start>                          <!-- Simstep to start sampling; default 0 -->
            <writefrequency>{cylSamplingFreq}</writefrequency>        <!-- Simstep to write out result file; default 10000 -->
            <stop>1000000000000</stop>                            <!-- Simstep to stop sampling; default 1000000000 -->
    </plugin>



	<plugin name="RegionSampling">
		<region>
			<coords>
				<lcx>{domainX/2 - 1}</lcx> <lcy refcoordsID="0">0.0</lcy> <lcz>{domainZ/2 - 1}</lcz>
				<ucx>{domainX/2 + 1}</ucx> <ucy refcoordsID="0">box</ucy> <ucz>{domainZ/2 + 1}</ucz>
			</coords>
			
			<sampling type="profiles">   <!-- Sampling profiles of various scalar and vector quantities, e.g. temperature, density, force, hydrodynamic velocity -->	
				<control>
					<start>0</start>           <!-- start time step -->
					<frequency>{rsfreq}</frequency>   <!-- frequency of writing profiles -->
					<stop>1000000000</stop>             <!-- stop time step -->
				</control>
				<subdivision type="number">       <!-- type="number | width" => subdivision of region into bins -->
					<width>10</width>         <!-- bin width -->
					<number>48</number>         <!-- number of bins -->
				</subdivision>
			</sampling>
		</region>
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
    #     elif arg == 'prod':
    #         execStep = 'prod'
    #     elif arg == 'test':
    #         work_folder = os.path.join(work_folder, f'test')  
    #     elif arg.startswith('m'):
    #         machine = int(eval(arg[1:]))
    #         if machine in [0,1,2]:
    #             print(f'argument {arg} interpreted as machine = {machine}')
    #         else:
    #             print(f'invalid machine argument [{machine}]. setting machine = 3')
    #             machine = 3
        # elif arg.startswith('T'):  
        #     if len(arg) < 3 or type(eval(arg[1:])) == type(1) or type(eval(arg[1:]))== type(.7): arg+=',' 
        #     TList= list(eval(arg[1:]))
        #     print(f'argument {arg} interpreted as TList = {TList}')
    #     elif arg.startswith('r'):
    #         if len(arg) < 3 or type(eval(arg[1:])) == type(1) or type(eval(arg[1:]))== type(.7): arg+=',' 
    #         rList= list(eval(arg[1:]))
    #         print(f'argument {arg} interpreted as rList = {rList}')
    #     else:
    #         print(f'arg {arg} not a valid argument')
    print(f'execstep: {execStep}')
    main()
