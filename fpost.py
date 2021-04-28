"""For reading FEHM output files."""

"""
Copyright 2013.
Los Alamos National Security, LLC. 
This material was produced under U.S. Government contract DE-AC52-06NA25396 for 
Los Alamos National Laboratory (LANL), which is operated by Los Alamos National 
Security, LLC for the U.S. Department of Energy. The U.S. Government has rights 
to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS 
ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES 
ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce 
derivative works, such modified software should be clearly marked, so as not to 
confuse it with the version available from LANL.

Additionally, this library is free software; you can redistribute it and/or modify 
it under the terms of the GNU Lesser General Public License as published by the 
Free Software Foundation; either version 2.1 of the License, or (at your option) 
any later version. Accordingly, this library is distributed in the hope that it 
will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General 
Public License for more details.
"""

import numpy as np
import os
try:
	from matplotlib import pyplot as plt
	from mpl_toolkits.mplot3d import axes3d
	from matplotlib import cm
	import matplotlib
except ImportError:
	'placeholder'
from copy import copy,deepcopy
from ftool import*
import platform

from fdflt import*
dflt = fdflt()
import pyvtk as pv

WINDOWS = platform.system()=='Windows'
if WINDOWS: copyStr = 'copy'; delStr = 'del'
else: copyStr = 'cp'; delStr = 'rm'

if True: 					# output variable dictionaries defined in here, indented for code collapse
	cont_var_names_avs=dict([
	('X coordinate (m)','x'),
	('Y coordinate (m)','y'),
	('Z coordinate (m)','z'),
	('node','n'),
	('Liquid Pressure (MPa)','P'),
	('Vapor Pressure (MPa)','P_vap'),
	('Capillary Pressure (MPa)','P_cap'),
	('Saturation','saturation'),
	('Temperature (deg C)','T'),
	('Porosity','por'),
	('X Permeability (log m**2)','perm_x'),
	('Y Permeability (log m**2)','perm_y'),
	('Z Permeability (log m**2)','perm_z'),
	('X displacement (m)','disp_x'),
	('Y displacement (m)','disp_y'),
	('Z displacement (m)','disp_z'),
	('X stress (MPa)','strs_xx'),
	('Y stress (MPa)','strs_yy'),
	('Z stress (MPa)','strs_zz'),
	('XY stress (MPa)','strs_xy'),
	('XZ stress (MPa)','strs_xz'),
	('YZ stress (MPa)','strs_yz'),
	('Youngs Mod (MPa)','E'),
	('Excess Shear (MPa)','tau_ex'),
	('Shear Angle (deg)','phi_dil'),
	('Zone','zone'),
	('Liquid Density (kg/m**3)','density'),
	('Vapor Density (kg/m**3)','density_vap'),
	('Source (kg/s)','flow'),
	('Liquid Flux (kg/s)','flux'),
	('Vapor Flux (kg/s)','flux_vap'),
	('Volume Strain','strain'),
	('Vapor X Volume Flux (m3/[m2 s])','flux_x_vap'),
	('Vapor Y Volume Flux (m3/[m2 s])','flux_y_vap'),
	('Vapor Z Volume Flux (m3/[m2 s])','flux_z_vap'),
	('Liquid X Volume Flux (m3/[m2 s])','flux_x'),
	('Liquid Y Volume Flux (m3/[m2 s])','flux_y'),
	('Liquid Z Volume Flux (m3/[m2 s])','flux_z'),
	]) 

	cont_var_names_tec=dict([
	('X coordinate (m)','x'),
	('Y coordinate (m)','y'),
	('Z coordinate (m)','z'),
	('X Coordinate (m)','x'),
	('Y Coordinate (m)','y'),
	('Z Coordinate (m)','z'),
	('node','n'),
	('Node','n'),
	('Liquid Pressure (MPa)','P'),
	('Vapor Pressure (MPa)','P_vap'),
	('Capillary Pressure (MPa)','P_cap'),
	('Saturation','saturation'),
	('Water Saturation','water'),
	('Super-Critical/Liquid CO2 Saturation','co2_sc_liquid'),
	('Gaseous CO2 Saturation','co2_gas'),
	('Dissolved CO2 Mass Fraction','co2_aq'),
	('CO2 Phase State','co2_phase'),
	('CO2 Gas Density (kg/m**3)','density_co2_gas'),
	('CO2 Liquid Density (kg/m**3)','density_co2_sc_liquid'),
	('Temperature (<sup>o</sup>C)','T'),
	('Temperature (deg C)','T'),
	('Porosity','por'),
	('X Permeability (log m**2)','perm_x'),
	('Y Permeability (log m**2)','perm_y'),
	('Z Permeability (log m**2)','perm_z'),
	('X displacement (m)','disp_x'),
	('Y displacement (m)','disp_y'),
	('Z displacement (m)','disp_z'),
	('X stress (MPa)','strs_xx'),
	('Y stress (MPa)','strs_yy'),
	('Z stress (MPa)','strs_zz'),
	('XY stress (MPa)','strs_xy'),
	('XZ stress (MPa)','strs_xz'),
	('YZ stress (MPa)','strs_yz'),
	('Youngs Mod (MPa)','E'),
	('Excess Shear (MPa)','tau_ex'),
	('Shear Angle (deg)','phi_dil'),
	('Zone','zone'),
	('Liquid Density (kg/m**3)','density'),
	('Vapor Density (kg/m**3)','density_vap'),
	('Source (kg/s)','flow'),
	('Liquid Flux (kg/s)','flux'),
	('Vapor Flux (kg/s)','flux_vap'),
	('Volume Strain','strain'),
	('Vapor X Volume Flux (m3/[m2 s])','flux_x_vap'),
	('Vapor Y Volume Flux (m3/[m2 s])','flux_y_vap'),
	('Vapor Z Volume Flux (m3/[m2 s])','flux_z_vap'),
	('Liquid X Volume Flux (m3/[m2 s])','flux_x'),
	('Liquid Y Volume Flux (m3/[m2 s])','flux_y'),
	('Liquid Z Volume Flux (m3/[m2 s])','flux_z'),
	])

	cont_var_names_surf=dict([
	('X coordinate (m)','x'), 
	('X Coordinate (m)','x'), 
	('X (m)','x'), 
	('Y coordinate (m)','y'), 
	('Y Coordinate (m)','y'), 
	('Y (m)','y'), 
	('Z coordinate (m)','z'), 
	('Z Coordinate (m)','z'), 
	('Z (m)','z'),
	('node','n'),
	('Node','n'),
	('Liquid Pressure (MPa)','P'),
	('Vapor Pressure (MPa)','P_vap'),
	('Capillary Pressure (MPa)','P_cap'),
	('Saturation','saturation'),
	('Temperature (deg C)','T'),
	('Porosity','por'),
	('X Permeability (log m**2)','perm_x'),
	('Y Permeability (log m**2)','perm_y'),
	('Z Permeability (log m**2)','perm_z'),
	('X displacement (m)','disp_x'),
	('Y displacement (m)','disp_y'),
	('Z displacement (m)','disp_z'),
	('X stress (MPa)','strs_xx'),
	('Y stress (MPa)','strs_yy'),
	('Z stress (MPa)','strs_zz'),
	('XY stress (MPa)','strs_xy'),
	('XZ stress (MPa)','strs_xz'),
	('YZ stress (MPa)','strs_yz'),
	('Youngs Mod (MPa)','E'),
	('Excess Shear (MPa)','tau_ex'),
	('Shear Angle (deg)','phi_dil'),
	('Zone','zone'),
	('Liquid Density (kg/m**3)','density'),
	('Vapor Density (kg/m**3)','density_vap'),
	('Source (kg/s)','flow'),
	('Liquid Flux (kg/s)','flux'),
	('Vapor Flux (kg/s)','flux_vap'),
	('Volume Strain','strain'),
	('Vapor X Volume Flux (m3/[m2 s])','flux_x_vap'),
	('Vapor Y Volume Flux (m3/[m2 s])','flux_y_vap'),
	('Vapor Z Volume Flux (m3/[m2 s])','flux_z_vap'),
	('Liquid X Volume Flux (m3/[m2 s])','flux_x'),
	('Liquid Y Volume Flux (m3/[m2 s])','flux_y'),
	('Liquid Z Volume Flux (m3/[m2 s])','flux_z'),
	('Water Saturation','saturation'),
	('Super-Critical/Liquid CO2 Saturation','co2_liquid'),
	('Gaseous CO2 Saturation','co2_gas'),
	('Dissolved CO2 Mass Fraction','co2_aq'),
	('CO2 Phase State','co2_phase'),
	('Aqueous_Species_001','Caq001'),
	('Aqueous_Species_002','Caq002'),
	('Aqueous_Species_003','Caq003'),
	('Aqueous_Species_004','Caq004'),
	('Aqueous_Species_005','Caq005'),
	('Aqueous_Species_006','Caq006'),
	('Aqueous_Species_007','Caq007'),
	('Aqueous_Species_008','Caq008'),
	('Aqueous_Species_009','Caq009'),
	('Aqueous_Species_010','Caq010'),
	('Aqueous_Species_011','Caq011'),
	('Aqueous_Species_012','Caq012'),
	('Aqueous_Species_013','Caq013'),
	('Aqueous_Species_014','Caq014'),
	('Aqueous_Species_015','Caq015'),
	('Aqueous_Species_016','Caq016'),
	('Aqueous_Species_017','Caq017'),
	('Aqueous_Species_018','Caq018'),
	('Aqueous_Species_019','Caq019'),
	('Aqueous_Species_020','Caq020'),
	])

	hist_var_names=dict([
	('denAIR','density_air'),
	('disx','disp_x'),
	('disy','disp_y'),
	('disz','disp_z'),
	('enth','enthalpy'),
	('glob','global'),
	('humd','humidity'),
	('satr','saturation'),
	('strain','strain'),
	('strx','strs_xx'),
	('stry','strs_yy'),
	('strz','strs_zz'),
	('strxy','strs_xy'),
	('strxz','strs_xz'),
	('stryz','strs_yz'),
	('wcon','water_content'),
	('denWAT','density'),
	('flow','flow'),
	('visAIR','viscosity_air'),
	('visWAT','viscosity'),
	('wt','water_table'),
	('presCAP','P_cap'),
	('presVAP','P_vap'),
	('presWAT','P'),
	('presCO2','P_co2'),
	('temp','T'),
	('co2md','massfrac_co2_aq'),
	('co2mf','massfrac_co2_free'),
	('co2mt','mass_co2'),
	('co2sg','saturation_co2g'),
	('co2sl','saturation_co2l'),
	])
	
	flxz_water_names = [
	'water_source',
	'water_sink',
	'water_net',
	'water_boundary',]
	
	flxz_vapor_names = [
	'vapor_source',
	'vapor_sink',
	'vapor_net',
	'vapor_boundary',]
	
	flxz_co2_names = [
	'co2_source',
	'co2_sink',
	'co2_in',
	'co2_out',
	'co2_boundary',
	'co2_sourceG',
	'co2_sinkG',
	'co2_inG',
	'co2_outG']
class fcontour(object): 					# Reading and plotting methods associated with contour output data.
	'''Contour output information object.
	
	'''
	def __init__(self,filename=None,latest=False,first=False,nearest=None):
		if not isinstance(filename,list):
			self._filename=os_path(filename)
		self._silent = dflt.silent
		self._times=[]   
		self._format = ''
		self._data={}
		self._material = {}
		self._material_properties = []
		self._row=None
		self._variables=[]  
		self._user_variables = []
		self.key_name=[]
		self._keyrows={}
		self.column_name=[]
		self.num_columns=0
		self._x = []
		self._y = []
		self._z = []
		self._xmin = None
		self._ymin = None
		self._zmin = None
		self._xmax = None
		self._ymax = None
		self._zmax = None
		self._latest = latest
		self._first = first
		self._nearest = nearest
		if isinstance(self._nearest,(float,int)): self._nearest = [self._nearest]
		self._nkeys=1
		if filename is not None: self.read(filename,self._latest,self._first,self._nearest)
	def __getitem__(self,key):
		if key in self.times:
			return self._data[key]
		elif np.min(abs(self.times-key)/self.times)<.01:
			ind = np.argmin(abs(self.times-key))
			return self._data[self.times[ind]]
		else: return None
	def read(self,filename,latest=False,first=False,nearest=[]): 						# read contents of file
		'''Read in FEHM contour output information.
		
		:param filename: File name for output data, can include wildcards to define multiple output files.
		:type filename: str
		:param latest: 	Boolean indicating PyFEHM should read the latest entry in a wildcard search.
		:type latest: bool
		:param first: Boolean indicating PyFEHM should read the first entry in a wildcard search.
		:type first: bool
		:param nearest: Read in the file with date closest to the day supplied. List input will parse multiple output files.
		:type nearest: fl64,list
		'''
		from glob import glob
		if isinstance(filename,list):
			files = filename
		else:
			filename = os_path(filename)
			files=glob(filename)
		if len(files)==0: 	
			pyfehm_print('ERROR: '+filename+' not found',self._silent)
			return
		# decision-making
		mat_file = None
		multi_type = None
		# are there multiple file types? e.g., _con_ and _sca_?
		# is there a material properties file? e.g., 'mat_nodes'?
		file_types = []
		for file in files:
			if '_sca_node' in file and 'sca' not in file_types: file_types.append('sca')
			if '_vec_node' in file and 'vec' not in file_types: file_types.append('vec')
			if '_con_node' in file and 'con' not in file_types: file_types.append('con')
			if '_hf_node' in file and 'hf' not in file_types: file_types.append('hf')
			if 'mat_node' in file: mat_file = file
		
		if self._nearest or latest or first:
			files = list(filter(os.path.isfile, glob(filename)))
			if mat_file: files.remove(mat_file)
			files.sort(key=lambda x: os.path.getmtime(x))
			files2 = []
			
			# retrieve first created and same time in group
			if first:
				files2.append(files[0])
				for file_type in file_types:
					tag = '_'+file_type+'_node'
					if tag in files2[-1]:
						prefix = files2[-1].split(tag)[0]
						break						
				for file in files:
					if file.startswith(prefix) and tag not in file: files2.append(file)
					
			# retrieve files nearest in time to given (and same time in group)
			if self._nearest:
				ts = []
				for file in files:
					file = file.split('_node')[0]
					file = file.split('_sca')[0]
					file = file.split('_con')[0]
					file = file.split('_vec')[0]
					file = file.split('_hf')[0]
					file = file.split('_days')[0]						
					file = file.split('.')
					file = file[-2:]
					ts.append(float('.'.join(file)))
				ts = np.unique(ts)
				
				for near in self._nearest:
					tsi = min(enumerate(ts), key=lambda x: abs(x[1]-near))[0]
					files2.append(files[tsi])
					for file_type in file_types:
						tag = '_'+file_type+'_node'
						if tag in files2[-1]:
							prefix = files2[-1].split(tag)[0]
							break
					for file in files:
						if file.startswith(prefix) and tag not in file: files2.append(file)						
						
			# retrieve last created and same time in group
			if latest:
				files2.append(files[-1])
				for file_type in file_types:
					tag = '_'+file_type+'_node'
					if tag in files2[-1]:
						prefix = files2[-1].split(tag)[0]
						break
				for file in files:
					if file.startswith(prefix) and tag not in file: files2.append(file)
					
			# removes duplicates
			files = []
			for file in files2:
				if file not in files: files.append(file)
		
		# group files into their types
		FILES = []
		for file_type in file_types:
			tag = '_'+file_type+'_node'
			FILES.append(sort_tec_files([file for file in files if tag in file]))
		FILES = np.array(FILES)
		
		# determine headers for 'tec' output
		for i in range(FILES.shape[1]):
			if not self._variables:
				files = FILES[:,i]
				headers = []
				for file in sort_tec_files(files):
					fp = open(file,'rU')
					headers.append(fp.readline())
					fp.close()
				firstFile = self._detect_format(headers)
				if self._format=='tec' and firstFile: 
					headers = []
					for file in sort_tec_files(files):
						fp = open(file,'rU')
						fp.readline()
						headers.append(fp.readline())
						fp.close()
					self._setup_headers_tec(headers)		
		
		# read in output data
		for i in range(FILES.shape[1]):
			files = FILES[:,i]
			# Skip -1 file if present
			if '-1' in files[0]: continue
			for file in sort_tec_files(files): pyfehm_print(file,self._silent)
			if not self._variables:
				headers = []
				for file in sort_tec_files(files):
					fp = open(file,'rU')
					headers.append(fp.readline())
					fp.close()
				self._detect_format(headers)
				#if self._format=='tec': self._setup_headers_tec(headers)
				if self._format=='avs': self._setup_headers_avs(headers,files)
				elif self._format=='avsx': self._setup_headers_avsx(headers)
				elif self._format=='surf': self._setup_headers_surf(headers)
				else: pyfehm_print('ERROR: Unrecognised format',self._silent);return
				self.num_columns = len(self.variables)+1
			if self.format == 'tec': self._read_data_tec(files,mat_file)
			elif self.format == 'surf': self._read_data_surf(files,mat_file)
			elif self.format == 'avs': self._read_data_avs(files,mat_file)
			elif self.format == 'avsx': self._read_data_avsx(files,mat_file)
		
		# assemble grid information
		if 'x' in self.variables:
			self._x = np.unique(self[self.times[0]]['x'])
			self._xmin,self._xmax = np.min(self.x), np.max(self.x)
		if 'y' in self.variables:
			self._y = np.unique(self[self.times[0]]['y'])
			self._ymin,self._ymax = np.min(self.y), np.max(self.y)
		if 'z' in self.variables:
			self._z = np.unique(self[self.times[0]]['z'])
			self._zmin,self._zmax = np.min(self.z), np.max(self.z)
		if dflt.parental_cont:
			print('')
			print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			print('WARNING:')
			print('')
			print('Contour data is indexed using the Pythonic convention in which the first index is 0. FEHM node numbering convention begins at 1.')
			print('')
			print('THEREFORE, to get the correct contour value for a particular node, you need to pass the node index MINUS 1. Using node index to access contour data will return incorrect values.')
			print('')
			print('For example:')
			print('>>> node10 = dat.grid.node[10]')
			print('>>> c = fcontour(\'*.csv\')')
			print('>>> T_node10 = c[c.times[-1]][\'T\'][node10.index - 1]')
			print('  or')
			print('>>> T_node10 = c[c.times[-1]][\'T\'][9]')
			print('will return the correct value for node 10.')
			print('')
			print('Do not turn off this message unless you understand how to correctly access nodal values from contour data.') 
			print('To turn off this message, open the environment file \'fdflt.py\' and set self.parental_cont = False')
			print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			print('')
	def _detect_format(self,headers):
		if headers[0].startswith('TITLE ='):		# check for TEC output
			self._format = 'tec'
		if headers[0].startswith('ZONE '):		# check for TEC output
			self._format = 'tec'
			return False
		elif headers[0].startswith('node, '):		# check for SURF output
			self._format = 'surf'
		elif headers[0].startswith('nodes at '):	# check for AVSX output
			self._format = 'avsx'
		elif headers[0].split()[0].isdigit():			# check for AVS output
			self._format = 'avs'
		return True
	def _setup_headers_avsx(self,headers): 		# headers for the AVSX output format
		self._variables.append('n')
		for header in headers:
			header = header.strip().split(' : ')
			for key in header[1:]: 
				if key in list(cont_var_names_avs.keys()):
					var = cont_var_names_avs[key]
				else: var = key
				self._variables.append(var)
	def _read_data_avsx(self,files,mat_file):				# read data in AVSX format
		datas = []
		for file in sorted(files): 		# use alphabetical sorting
			fp = open(file,'rU')
			header = fp.readline()
			if file == sorted(files)[0]:
				header = header.split('nodes at ')[1]
				header = header.split('days')[0]
				time = float(header)*24*2600
				self._times.append(time)		
			lns = fp.readlines()
			fp.close()
			datas.append(np.array([[float(d) for d in ln.strip().split(':')] for ln in lns]))
		data = np.concatenate(datas,1)
		self._data[time] = dict([(var,data[:,icol]) for icol,var in enumerate(self.variables)])
		
		if mat_file and not self._material_properties:
			fp = open(mat_file,'rU')
			header = fp.readline()
			self._material_properties = header.split(':')[1:]
			lns = fp.readlines()
			fp.close()
			data = np.array([[float(d) for d in ln.strip().split(':')[1:]] for ln in lns])
			self._material= dict([(var,data[:,icol]) for icol,var in enumerate(self._material_properties)])
	def _setup_headers_avs(self,headers,files): 		# headers for the AVS output format
		for header,file in zip(headers,files):
			lns_num = int(header.strip().split()[0])
			fp = open(file)
			lns = [fp.readline() for i in range(lns_num+1)][1:]
			fp.close()
			self._variables.append('n')
			for ln in lns:
				varname = ln.strip().split(',')[0]
				if varname in list(cont_var_names_avs.keys()):
					var = cont_var_names_avs[varname]
				else: var = varname
				if var not in self._variables: self._variables.append(var)
	def _read_data_avs(self,files,mat_file):		# read data in AVS format
		datas = []
		for file in sorted(files):
			first = (file == sorted(files)[0])
			fp = open(file,'rU')
			lns = fp.readlines()
			lns = lns[int(float(lns[0].split()[0]))+1:]
			
			if first: 
				file = file.split('_node')[0]
				file = file.split('_sca')[0]
				file = file.split('_con')[0]
				file = file.split('_vec')[0]
				file = file.split('_hf')[0]
				file = file.split('_days')[0]						
				file = file.split('.')
				file = [fl for fl in file if fl.isdigit() or 'E-' in fl]
				time = float('.'.join(file))
				self._times.append(time)
			
			if first: 
				datas.append(np.array([[float0(d) for d in ln.strip().split()] for ln in lns]))
			else:
				datas.append(np.array([[float0(d) for d in ln.strip().split()[4:]] for ln in lns]))
			
		data = np.concatenate(datas,1)
		self._data[time] = dict([(var,data[:,icol]) for icol,var in enumerate(self.variables)])
	def _setup_headers_surf(self,headers): 		# headers for the SURF output format
		for header in headers:
			header = header.strip().split(', ')
			for key in header: 
				varname = key.split('"')[0]
				varname = varname.strip()
				if varname in list(cont_var_names_surf.keys()):
					var = cont_var_names_surf[varname]
				else: var = varname
				if var not in self._variables: self._variables.append(var)
	def _read_data_surf(self,files,mat_file):		# read data in SURF format
		datas = []
		for file in sorted(files):
			first = (file == sorted(files)[0])
			fp = open(file,'rU')
			lni = file.split('.',1)[1]
			
			if first: 
				file = file.split('_node')[0]
				file = file.split('_sca')[0]
				file = file.split('_con')[0]
				file = file.split('_vec')[0]
				file = file.split('_hf')[0]
				file = file.split('_days')[0]						
				file = file.split('.')
				file = [fl for fl in file if fl.isdigit() or 'E-' in fl]
				time = float('.'.join(file))
				self._times.append(time)
			
			lni=fp.readline()			
			lns = fp.readlines()
			fp.close()
			
			if first: 
				datas.append(np.array([[float0(d) for d in ln.strip().split(',')] for ln in lns]))
			else:
				datas.append(np.array([[float0(d) for d in ln.strip().split(',')[4:]] for ln in lns]))
			
		data = np.concatenate(datas,1)
		self._data[time] = dict([(var,data[:,icol]) for icol,var in enumerate(self.variables)])
		
		if mat_file and not self._material_properties:
			fp = open(mat_file,'rU')
			header = fp.readline()
			for mat_prop in header.split(',')[1:]:
				if 'specific heat' not in mat_prop:
					self._material_properties.append(mat_prop.strip())
			lns = fp.readlines()
			fp.close()
			data = np.array([[float(d) for d in ln.strip().split(',')[1:]] for ln in lns])
			self._material= dict([(var,data[:,icol]) for icol,var in enumerate(self._material_properties)])
	def _setup_headers_tec(self,headers): 		# headers for the TEC output format
		for header in headers:
			header = header.split(' "')
			for key in header[1:]: 
				varname = key.split('"')[0].strip()
				if varname in list(cont_var_names_tec.keys()):
					var = cont_var_names_tec[varname]
				else: var = varname
				
				if var not in self._variables: self._variables.append(var)
	def _read_data_tec(self,files,mat_file):						# read data in TEC format
		datas = []
		for file in sorted(files):
			first = (file == sorted(files)[0])
			fp = open(file,'rU')
			ln = fp.readline()
			has_xyz = False
			while not ln.startswith('ZONE'):
				ln = fp.readline()
				has_xyz = True
			
			if first: 
				lni = ln.split('"')[1]
				time = lni.split('days')[0].strip()
				time = float(time.split()[-1].strip())
				try:
					if time<self._times[0]: return
				except: pass
				self._times.append(time)
				nds = None
				if 'N =' in ln: 
					nds = int(ln.split('N = ')[-1].strip().split(',')[0].strip())
			
			lns = fp.readlines()
			fp.close()
			if nds: lns = lns[:nds] 		# truncate to remove connectivity information
			
			if has_xyz:
				if first: 
					datas.append(np.array([[float0(d) for d in ln.strip().split()] for ln in lns]))
				else:
					datas.append(np.array([[float0(d) for d in ln.strip().split()[4:]] for ln in lns]))
			else:
				if first: 
					datas.append(np.array([[float0(d) for d in ln.strip().split()] for ln in lns]))
				else:
					datas.append(np.array([[float0(d) for d in ln.strip().split()[1:]] for ln in lns]))
						
		data = np.concatenate(datas,1)
		if data.shape[1]< len(self.variables): 		# insert xyz data from previous read
			data2 = []
			j = 0
			for var in self.variables:
				if var == 'x': 
					data2.append(self._data[self._times[0]]['x'])
				elif var == 'y': 
					data2.append(self._data[self._times[0]]['y'])
				elif var == 'z': 
					data2.append(self._data[self._times[0]]['z'])
				elif var == 'zone': 
					data2.append(self._data[self._times[0]]['zone'])
				else: 
					data2.append(data[:,j]); j +=1
			data = np.transpose(np.array(data2))
		self._data[time] = dict([(var,data[:,icol]) for icol,var in enumerate(self.variables)])
		if mat_file and not self._material_properties:
			fp = open(mat_file,'rU')
			fp.readline()
			header = fp.readline()
			for mat_prop in header.split(' "')[5:]:
				self._material_properties.append(mat_prop.split('"')[0].strip())
			lns = fp.readlines()
			if lns[0].startswith('ZONE'): lns = lns[1:]
			fp.close()
			if nds: lns = lns[:nds] 		# truncate to remove connectivity information
			data = np.array([[float(d) for d in ln.strip().split()[4:]] for ln in lns[:-1]])
			self._material= dict([(var,data[:,icol]) for icol,var in enumerate(self._material_properties)])
	def _check_inputs(self,variable, time, slice):	# assesses whether sufficient input information for slice plot
		if not variable: 
			s = ['ERROR: no plot variable specified.']
			s.append('Options are')
			for var in self.variables: s.append(var)
			s = '\n'.join(s)
			pyfehm_print(s,self._silent)
			return True
		if time==None: 
			s = ['ERROR: no plot time specified.']
			s.append('Options are')
			for time in self.times: s.append(time)
			s = '\n'.join(s)
			pyfehm_print(s,self._silent)
			return True
		if not slice: 
			s = ['Error: slice orientation undefined.']
			s.append('Options are')
			s.append('[\'x\',float] - slice parallel to y-axis at x=float')
			s.append('[\'y\',float] - slice parallel to x-axis at y=float')
			s.append('[\'theta\',float] - angle measured anti-clockwise from +x')
			s.append('[[float,float],[float,float]] - point to point')
			s = '\n'.join(s)
			pyfehm_print(s,self._silent)
			return True
		return False
	def new_variable(self,name,time,data): 	
		'''Creates a new variable, which is some combination of the available variables.
		
		:param name: Name for the variable.
		:type name: str
		:param time: Time key which the variable should be associated with. Must be one of the existing keys, i.e., an item in fcontour.times.
		:type time: fl64
		:param data: Variable data, most likely some combination of the available parameters, e.g., pressure*temperature, pressure[t=10] - pressure[t=5]
		:type data: lst[fl64]
		'''
		if time not in self.times: 
			pyfehm_print('ERROR: supplied time must correspond to an existing time in fcontour.times',self._silent)
			return
		if name in self.variables:
			pyfehm_print('ERROR: there is already a variable called \''+name+'\', please choose a different name',self._silent)
			return
		self._data[time][name] = data
		if name not in self._user_variables:
			self._user_variables.append(name)
	def slice(self, variable, slice, divisions, time=None, method='nearest'):
		'''Returns mesh data for a specified slice orientation from 3-D contour output data.
		
		:param variable: Output data variable, for example 'P' = pressure. Alternatively, variable can be a five element list, first element 'cfs', remaining elements fault azimuth (relative to x), dip, friction coefficient and cohesion. Will return coulomb failure stress.
		:type variable: str
		:param time: Time for which output data is requested. Can be supplied via ``fcontour.times`` list. Default is most recently available data.
		:type time: fl64
		:param slice: List specifying orientation of output slice, e.g., ['x',200.] is a vertical slice at ``x = 200``, ['z',-500.] is a horizontal slice at ``z = -500.``, [point1, point2] is a fixed limit vertical or horizontal domain corresponding to the bounding box defined by point1 and point2.
		:type slice: lst[str,fl64]
		:param divisions: Resolution to supply mesh data.
		:type divisions: [int,int]
		:param method: Method of interpolation, options are 'nearest', 'linear'.
		:type method: str:	
		:returns: X -- x-coordinates of mesh data.
		
		'''
		if time==None: 
			if np.min(self.times)<0: time = self.times[0]
			else: time = self.times[-1]
		from scipy.interpolate import griddata		
		delta = False
		if isinstance(time,list) or isinstance(time,np.ndarray):
			if len(time)>1: 
				time0 = np.min(time)
				time = np.max(time)
				delta=True		
		dat = self[time]
		
		# check to see if cfs plot requested
		cfs = False
		if isinstance(variable,list):
			if variable[0] in ['cfs','CFS']: cfs = True
		
		if not cfs:
			if delta: dat0 = self[time0]
			if isinstance(slice[0],str):
				if slice[0].startswith('y'):
					xmin = np.min(dat['x']);xmax = np.max(dat['x'])
					ymin = np.min(dat['z']);ymax = np.max(dat['z'])		
					if slice[1] is None:
						points = np.transpose(np.array([dat['x'],dat['z'],np.ones((1,len(dat['z'])))[0]]))
						slice[1] = 1
					else:
						points = np.transpose(np.array([dat['x'],dat['z'],dat['y']]))
				elif slice[0].startswith('x'):
					xmin = np.min(dat['y']);xmax = np.max(dat['y'])
					ymin = np.min(dat['z']);ymax = np.max(dat['z'])		
					if slice[1] is None:
						points = np.transpose(np.array([dat['y'],dat['z'],np.ones((1,len(dat['z'])))[0]]))
						slice[1] = 1
					else:
						points = np.transpose(np.array([dat['y'],dat['z'],dat['x']]))
				elif slice[0].startswith('z'):
					xmin = np.min(dat['x']);xmax = np.max(dat['x'])
					ymin = np.min(dat['y']);ymax = np.max(dat['y'])		
					if slice[1] is None:
						points = np.transpose(np.array([dat['x'],dat['y'],np.ones((1,len(dat['y'])))[0]]))
						slice[1] = 1
					else:
						points = np.transpose(np.array([dat['x'],dat['y'],dat['z']]))
				elif slice[0].startswith('theta'): 
					pyfehm_print('ERROR: theta slicing not supported yet',self._silent)	
					return
				xrange = np.linspace(xmin,xmax,divisions[0])
				yrange = np.linspace(ymin,ymax,divisions[1])
				X,Y = np.meshgrid(xrange,yrange)
				Z = (X+np.sqrt(1.757))/(X+np.sqrt(1.757))*slice[1]
				pointsI = np.transpose(np.reshape((X,Y,Z),(3,X.size)))
				vals = np.transpose(np.array(dat[variable]))
				valsI = griddata(points,vals,pointsI,method=method)
				valsI =  np.reshape(valsI,(X.shape[0],X.shape[1]))
				if delta:
					vals = np.transpose(np.array(dat0[variable]))
					valsI0 = griddata(points,vals,pointsI,method=method)
					valsI0 =  np.reshape(valsI0,(X.shape[0],X.shape[1]))
					valsI = valsI - valsI0
			elif isinstance(slice[0],list):
				# check if horizontal or vertical slice
				dx,dy,dz = abs(slice[0][0]-slice[1][0]),abs(slice[0][1]-slice[1][1]),abs(slice[0][2]-slice[1][2])
				if 100*dz<dx and 100*dz<dy: 	#horizontal
					xmin,xmax = np.min([slice[0][0],slice[1][0]]),np.max([slice[0][0],slice[1][0]])
					ymin,ymax = np.min([slice[0][1],slice[1][1]]),np.max([slice[0][1],slice[1][1]])
					xrange = np.linspace(xmin,xmax,divisions[0])
					yrange = np.linspace(ymin,ymax,divisions[1])
					X,Y = np.meshgrid(xrange,yrange)
					Z = (X+np.sqrt(1.757))/(X+np.sqrt(1.757))*(slice[0][2]+slice[1][2])/2
				else: 							#vertical 
					xmin,xmax = 0,np.sqrt((slice[0][0]-slice[1][0])**2+(slice[0][1]-slice[1][1])**2)
					ymin,ymax = np.min([slice[0][2],slice[1][2]]),np.max([slice[0][2],slice[1][2]])
					xrange = np.linspace(xmin,xmax,divisions[0])
					yrange = np.linspace(ymin,ymax,divisions[1])
					X,Z = np.meshgrid(xrange,yrange)
					Y = X/xmax*abs(slice[0][1]-slice[1][1]) + slice[0][1]
					X = X/xmax*abs(slice[0][0]-slice[1][0]) + slice[0][0]
				points = np.transpose(np.array([dat['x'],dat['y'],dat['z']]))
				pointsI = np.transpose(np.reshape((X,Y,Z),(3,X.size)))
				vals = np.transpose(np.array(dat[variable]))
				valsI = griddata(points,vals,pointsI,method=method)
				valsI =  np.reshape(valsI,(X.shape[0],X.shape[1]))
				if delta:
					vals = np.transpose(np.array(dat0[variable]))
					valsI0 = griddata(points,vals,pointsI,method=method)
					valsI0 =  np.reshape(valsI0,(X.shape[0],X.shape[1]))
					valsI = valsI - valsI0
			
		else:
			if delta: time0 = time[0]; time = time[-1]
			X,Y,Z,sxx = self.slice('strs_xx', slice, divisions, time, method)
			X,Y,Z,syy = self.slice('strs_yy', slice, divisions, time, method)
			X,Y,Z,szz = self.slice('strs_zz', slice, divisions, time, method)
			X,Y,Z,sxy = self.slice('strs_xy', slice, divisions, time, method)
			X,Y,Z,sxz = self.slice('strs_xz', slice, divisions, time, method)
			X,Y,Z,syz = self.slice('strs_yz', slice, divisions, time, method)
			X,Y,Z,sp  = self.slice('P',	   slice, divisions, time, method)
			
			dip = variable[2]/180.*math.pi
			azi = variable[1]/180.*math.pi+3.14159/2.
			nhat = np.array([np.cos(azi)*np.sin(dip),np.sin(azi)*np.sin(dip),np.cos(dip)])
			mu = variable[3]
			cohesion = variable[4]
			
			px = sxx*nhat[0]+sxy*nhat[1]+sxz*nhat[2]
			py = sxy*nhat[0]+syy*nhat[1]+syz*nhat[2]
			pz = sxz*nhat[0]+syz*nhat[1]+szz*nhat[2]
			
			sig = px*nhat[0]+py*nhat[1]+pz*nhat[2]
			tau = np.sqrt(px**2+py**2+pz**2 - sig**2)
			valsI = tau - mu*(sig-sp) - cohesion
			if delta:
				X,Y,Z,sxx = self.slice('strs_xx', slice, divisions, time0, method)
				X,Y,Z,syy = self.slice('strs_yy', slice, divisions, time0, method)
				X,Y,Z,szz = self.slice('strs_zz', slice, divisions, time0, method)
				X,Y,Z,sxy = self.slice('strs_xy', slice, divisions, time0, method)
				X,Y,Z,sxz = self.slice('strs_xz', slice, divisions, time0, method)
				X,Y,Z,syz = self.slice('strs_yz', slice, divisions, time0, method)
				X,Y,Z,sp  = self.slice('P',	   slice, divisions, time0, method)
				
				px = sxx*nhat[0]+sxy*nhat[1]+sxz*nhat[2]
				py = sxy*nhat[0]+syy*nhat[1]+syz*nhat[2]
				pz = sxz*nhat[0]+syz*nhat[1]+szz*nhat[2]
				
				sig = px*nhat[0]+py*nhat[1]+pz*nhat[2]
				tau = np.sqrt(px**2+py**2+pz**2 - sig**2)
				valsI = valsI - (tau - mu*(sig-sp) - cohesion)
				
		return X, Y, Z, valsI
	def slice_plot_line(self,variable=None,time=None,slice='',divisions=[20,20],labels=False, label_size=10.,levels=10,xlims=[],	
		ylims=[],colors='k',linestyle='-',save='',	xlabel='x / m',ylabel='y / m',title='', font_size='medium', method='nearest',
		equal_axes=True):	
		'''Returns a line plot of contour data. Invokes the ``slice()`` method to interpolate slice data for plotting.
		
		:param variable: Output data variable, for example 'P' = pressure.
		:type variable: str
		:param time: Time for which output data is requested. Can be supplied via ``fcontour.times`` list. Default is most recently available data. If a list of two times is passed, the difference between the first and last is plotted.
		:type time: fl64
		:param slice: List specifying orientation of output slice, e.g., ['x',200.] is a vertical slice at ``x = 200``, ['z',-500.] is a horizontal slice at ``z = -500.``, [point1, point2] is a fixed limit vertical or horizontal domain corresponding to the bounding box defined by point1 and point2.
		:type slice: lst[str,fl64]
		:param divisions: Resolution to supply mesh data.
		:type divisions: [int,int]
		:param method: Method of interpolation, options are 'nearest', 'linear'.
		:type method: str
		:param labels: Specify whether labels should be added to contour plot.
		:type labels: bool
		:param label_size: Specify text size of labels on contour plot, either as an integer or string, e.g., 10, 'small', 'x-large'.
		:type label_size: str, int
		:param levels: Contour levels to plot. Can specify specific levels in list form, or a single integer indicating automatic assignment of levels. 
		:type levels: lst[fl64], int
		:param xlims: Plot limits on x-axis.
		:type xlims: [fl64, fl64]
		:param ylims: Plot limits on y-axis.
		:type ylims: [fl64, fl64]
		:param linestyle: Style of contour lines, e.g., 'k-' = solid black line, 'r:' red dotted line.
		:type linestyle: str
		:param save: Name to save plot. Format specified extension (default .png if none give). Supported extensions: .png, .eps, .pdf.
		:type save: str
		:param xlabel: Label on x-axis.
		:type xlabel: str
		:param ylabel: Label on y-axis.
		:type ylabel: str
		:param title: Plot title.
		:type title: str
		:param font_size: Specify text size, either as an integer or string, e.g., 10, 'small', 'x-large'.
		:type font_size: str, int
		:param equal_axes: Specify equal scales on axes.
		:type equal_axes: bool
		
		'''	
		save = os_path(save)
		# at this stage, only structured grids are supported
		if time==None: time = self.times[-1]
		delta = False
		if isinstance(time,list) or isinstance(time,np.ndarray):
			if len(time)>1: 
				time0 = np.min(time)
				time = np.max(time)
				delta=True
		return_flag = self._check_inputs(variable,time,slice)
		if return_flag: return
		# gather plot data
		X, Y, Z, valsI = self.slice(variable=variable, time=time, slice=slice, divisions=divisions, method=method)
		if delta:
			X, Y, Z, valsIi = self.slice(variable=variable, time=time0, slice=slice, divisions=divisions, method=method)
			valsI = valsI - valsIi
		plt.clf()
		plt.figure(figsize=[8,8])
		ax = plt.axes([0.15,0.15,0.75,0.75])
		if xlims: ax.set_xlim(xlims)
		if ylims: ax.set_ylim(ylims)
		if equal_axes: ax.set_aspect('equal', 'datalim')
		CS = plt.contour(X,Y,valsI,levels,colors=colors,linestyle=linestyle)
		if labels: plt.clabel(CS,incline=1,fontsize=label_size)
		if xlabel: plt.xlabel(xlabel,size=font_size)		
		if ylabel: plt.ylabel(ylabel,size=font_size)
		if title: plt.title(title,size=font_size)
		for t in ax.get_xticklabels():
			t.set_fontsize(font_size)
		for t in ax.get_yticklabels():
			t.set_fontsize(font_size)
					
		extension, save_fname, pdf = save_name(save,variable=variable,time=time)
		plt.savefig(save_fname, dpi=100, facecolor='w', edgecolor='w',orientation='portrait', 
		format=extension,transparent=True, bbox_inches=None, pad_inches=0.1)
		if pdf: 
			os.system('epstopdf ' + save_fname)
			os.remove(save_fname)	
	def slice_plot(self,variable=None,time=None,slice='',divisions=[20,20],levels=10,cbar=False,xlims=[],
		ylims=[],colors='k',linestyle='-',save='',	xlabel='x / m',ylabel='y / m',title='', font_size='medium', method='nearest',
		equal_axes=True,mesh_lines = None,perm_contrasts=None, scale = 1.):		
		'''Returns a filled plot of contour data. Invokes the ``slice()`` method to interpolate slice data for plotting.
		
		:param variable: Output data variable, for example 'P' = pressure.
		:type variable: str
		:param time: Time for which output data is requested. Can be supplied via ``fcontour.times`` list. Default is most recently available data. If a list of two times is passed, the difference between the first and last is plotted.
		:type time: fl64
		:param slice: List specifying orientation of output slice, e.g., ['x',200.] is a vertical slice at ``x = 200``, ['z',-500.] is a horizontal slice at ``z = -500.``, [point1, point2] is a fixed limit vertical or horizontal domain corresponding to the bounding box defined by point1 and point2.
		:type slice: lst[str,fl64]
		:param divisions: Resolution to supply mesh data.
		:type divisions: [int,int]
		:param method: Method of interpolation, options are 'nearest', 'linear'.
		:type method: str
		:param levels: Contour levels to plot. Can specify specific levels in list form, or a single integer indicating automatic assignment of levels. 
		:type levels: lst[fl64], int
		:param cbar: Add colour bar to plot.
		:type cbar: bool
		:param xlims: Plot limits on x-axis.
		:type xlims: [fl64, fl64]
		:param ylims: Plot limits on y-axis.
		:type ylims: [fl64, fl64]
		:param colors: Specify colour string for contour levels.
		:type colors: lst[str]
		:param linestyle: Style of contour lines, e.g., 'k-' = solid black line, 'r:' red dotted line.
		:type linestyle: str
		:param save: Name to save plot. Format specified extension (default .png if none give). Supported extensions: .png, .eps, .pdf.
		:type save: str
		:param xlabel: Label on x-axis.
		:type xlabel: str
		:param ylabel: Label on y-axis.
		:type ylabel: str
		:param title: Plot title.
		:type title: str
		:param font_size: Specify text size, either as an integer or string, e.g., 10, 'small', 'x-large'.
		:type font_size: str, int
		:param equal_axes: Specify equal scales on axes.
		:type equal_axes: bool
		:param mesh_lines: Superimpose mesh on the plot (line intersections correspond to node positions) according to specified linestyle, e.g., 'k:' is a dotted black line.
		:type mesh_lines: bool
		:param perm_contrasts: Superimpose permeability contours on the plot according to specified linestyle, e.g., 'k:' is a dotted black line. A gradient method is used to pick out sharp changes in permeability.
		:type perm_contrasts: bool
		
		'''	
		if save:
			save = os_path(save)
		# at this stage, only structured grids are supported
		if time==None: 
			if np.min(self.times)<0: time = self.times[0]
			else: time = self.times[-1]
		delta = False
		if isinstance(time,list) or isinstance(time,np.ndarray):
			if len(time)>1: 
				time0 = np.min(time)
				time = np.max(time)
				delta=True
		# if data not available for one coordinate, assume 2-D simulation, adjust slice accordingly
		if 'x' not in self.variables: slice = ['x',None]
		if 'y' not in self.variables: slice = ['y',None]
		if 'z' not in self.variables: slice = ['z',None]
		return_flag = self._check_inputs(variable=variable,time=time,slice=slice)
		if return_flag: return
		# gather plot data
		X, Y, Z, valsI = self.slice(variable=variable, time=time, slice=slice, divisions=divisions, method=method)
		if delta:
			X, Y, Z, valsIi = self.slice(variable=variable, time=time0, slice=slice, divisions=divisions, method=method)
			valsI = valsI - valsIi
		if isinstance(slice[0],list):
			# check if horizontal or vertical slice
			dx,dy,dz = abs(slice[0][0]-slice[1][0]),abs(slice[0][1]-slice[1][1]),abs(slice[0][2]-slice[1][2])
			if not(100*dz<dx and 100*dz<dy): 	
				if dx < dy:
					X = Y
				Y = Z
					
		plt.clf()
		plt.figure(figsize=[8,8])
		ax = plt.axes([0.15,0.15,0.75,0.75])
		if xlims: ax.set_xlim(xlims)
		if ylims: ax.set_ylim(ylims)
		if equal_axes: ax.set_aspect('equal', 'datalim')
		if not isinstance(scale,list):
			CS = plt.contourf(X,Y,valsI*scale,levels)
		elif len(scale) == 2:
			CS = plt.contourf(X,Y,valsI*scale[0]+scale[1],levels)
		if xlabel: plt.xlabel(xlabel,size=font_size)		
		if ylabel: plt.ylabel(ylabel,size=font_size)
		if title: plt.title(title,size=font_size)
		if cbar:			
			cbar=plt.colorbar(CS)
			for t in cbar.ax.get_yticklabels():
				t.set_fontsize(font_size)
		for t in ax.get_xticklabels():
			t.set_fontsize(font_size)
		for t in ax.get_yticklabels():
			t.set_fontsize(font_size)
			
		if perm_contrasts:
			if 'perm_x' not in self.variables: 
				pyfehm_print('WARNING: No permeability data to construct unit boundaries.',self._silent)
			else:
				X, Y, Z, k = self.slice(variable='perm_z', time=time, slice=slice, divisions=divisions, method=method)
				# calculate derivatives in X and Y directions
				dkdX = np.diff(k,1,0)#/np.diff(Y,1,0)
				dkdX = (dkdX[1:,1:-1]+dkdX[:-1,1:-1])/2
				dkdY = np.diff(k,1,1)#/np.diff(X,1,1)
				dkdY = (dkdY[1:-1,1:]+dkdY[1:-1,:-1])/2
				dk = (abs((dkdX+dkdY)/2)>0.2)*1.
				col = 'k'; ln = '-'
				for let in perm_contrasts:
					if let in ['k','r','g','b','m','c','y','w']: col = let
					if let in ['-','--','-.',':']: ln = let
				CS = plt.contour(X[1:-1,1:-1],Y[1:-1,1:-1],dk,[0.99999999999],colors=col,linestyles=ln)
				
		xlims = ax.get_xlim()
		ylims = ax.get_ylim()
		if mesh_lines:
			# add grid lines
			ax.set_xlim(xlims[0],xlims[1])
			ax.set_ylim(ylims[0],ylims[1])
			if slice[0] == 'z':
				for t in np.unique(self[self.times[0]]['x']):
					ax.plot([t,t],[ylims[0],ylims[1]],mesh_lines,zorder=100)
				for t in np.unique(self[self.times[0]]['y']):
					ax.plot([xlims[0],xlims[1]],[t,t],mesh_lines,zorder=100)
			elif slice[0] == 'x':
				for t in np.unique(self[self.times[0]]['y']):
					ax.plot([t,t],[ylims[0],ylims[1]],mesh_lines,zorder=100)
				for t in np.unique(self[self.times[0]]['z']):
					ax.plot([xlims[0],xlims[1]],[t,t],mesh_lines,zorder=100)
			elif slice[0] == 'y':
				for t in np.unique(self[self.times[0]]['x']):
					ax.plot([t,t],[ylims[0],ylims[1]],mesh_lines,zorder=100)
				for t in np.unique(self[self.times[0]]['z']):
					ax.plot([xlims[0],xlims[1]],[t,t],mesh_lines,zorder=100)
		
		if save:
			extension, save_fname, pdf = save_name(save=save,variable=variable,time=time)
			plt.savefig(save_fname, dpi=100, facecolor='w', edgecolor='w',orientation='portrait', 
			format=extension,transparent=True, bbox_inches=None, pad_inches=0.1)
			if pdf: 
				os.system('epstopdf ' + save_fname)
				os.remove(save_fname)	
		else:
			plt.show()
	def profile(self, variable, profile, time=None, divisions=30, method='nearest'):
		'''Return variable data along the specified line in 3-D space. If only two points are supplied,
		the profile is assumed to be a straight line between them.
		
		:param variable: Output data variable, for example 'P' = pressure. Can specify multiple variables with a list.
		:type variable: str, lst[str]
		:param time: Time for which output data is requested. Can be supplied via ``fcontour.times`` list. Default is most recently available data.
		:type time: fl64
		:param profile: Three column array with each row corresponding to a point in the profile.
		:type profile: ndarray
		:param divisions: Number of points in profile. Only relevant if straight line profile being constructed from two points.
		:type divisions: int
		:param method: Interpolation method, options are 'nearest' (default) and 'linear'.
		:type method: str
		
		:returns: Multi-column array. Columns are in order x, y and z coordinates of profile, followed by requested variables.
		
		'''
		if isinstance(profile,list): profile = np.array(profile)
		if divisions: divisions = int(divisions)
		if time==None: time = self.times[-1]		
		from scipy.interpolate import griddata
		if not isinstance(variable,list): variable = [variable,]
		
		dat = self[time]
		points = np.transpose(np.array([dat['x'],dat['y'],dat['z']]))
		
		if profile.shape[0]==2:
			# construct line profile		
			xrange = np.linspace(profile[0][0],profile[1][0],divisions)
			yrange = np.linspace(profile[0][1],profile[1][1],divisions)
			zrange = np.linspace(profile[0][2],profile[1][2],divisions)
			profile = np.transpose(np.array([xrange,yrange,zrange]))
		
		outpoints = [list(profile[:,0]),list(profile[:,1]),list(profile[:,2])]
		for var in variable:
			vals = np.transpose(np.array(dat[var]))
			valsI = griddata(points,vals,profile,method=method)
			outpoints.append(list(valsI))
		
		return np.array(outpoints).transpose()
	def profile_plot(self,variable=None,time=None, profile=[],divisions = 30,xlims=[],ylims=[],
		color='k',marker='x-',save='',xlabel='distance / m',ylabel='',title='',font_size='medium',method='nearest',
		verticalPlot=False,elevationPlot=False):
		'''Return a plot of the given variable along a specified profile. If the profile comprises two points, 
		these are interpreted as the start and end points of a straight line profile.
		
		:param variable: Output data variable, for example 'P' = pressure. Can specify multiple variables with a list.
		:type variable: str, lst[str]
		:param time: Time for which output data is requested. Can be supplied via ``fcontour.times`` list. Default is most recently available data. If a list of two times is passed, the difference between the first and last is plotted.
		:type time: fl64
		:param profile: Three column array with each row corresponding to a point in the profile.
		:type profile: ndarray
		:param divisions: Number of points in profile. Only relevant if straight line profile being constructed from two points.
		:type divisions: int
		:param method: Interpolation method, options are 'nearest' (default) and 'linear'.
		:type method: str
		:param xlims: Plot limits on x-axis.
		:type xlims: [fl64, fl64]
		:param ylims: Plot limits on y-axis.
		:type ylims: [fl64, fl64]
		:param color: Colour of profile.
		:type color: str
		:param marker: Style of line, e.g., 'x-' = solid line with crosses, 'o:' dotted line with circles.
		:type marker: str
		:param save: Name to save plot. Format specified extension (default .png if none give). Supported extensions: .png, .eps, .pdf.
		:type save: str
		:param xlabel: Label on x-axis.
		:type xlabel: str
		:param ylabel: Label on y-axis.
		:type ylabel: str
		:param title: Plot title.
		:type title: str
		:param font_size: Specify text size, either as an integer or string, e.g., 10, 'small', 'x-large'.
		:type font_size: str, int
		
		:param verticalPlot: Flag to plot variable against profile distance on the y-axis.
		:type verticalPlot: bool
		:param elevationPlot: Flag to plot variable against elevation on the y-axis.
		:type elevationPlot: bool
		
		'''
		save = os_path(save)
		if time==None: time = self.times[-1]	
		delta = False
		if isinstance(time,list) or isinstance(time,np.ndarray):
			if len(time)>1: 
				time0 = np.min(time)
				time = np.max(time)
				delta=True
		if not variable: 
			s = ['ERROR: no plot variable specified.']
			s.append('Options are')
			for var in self.variables: s.append(var)
			s = '\n'.join(s)
			pyfehm_print(s,self._silent)
			return True
		if not ylabel: ylabel = variable
		
		plt.clf()
		plt.figure(figsize=[8,8])
		ax = plt.axes([0.15,0.15,0.75,0.75])
		outpts = self.profile(variable=variable,profile=profile,time=time,divisions=divisions,method=method)
		if delta:
			outptsI = self.profile(variable=variable,profile=profile,time=time,divisions=divisions,method=method)
			outpts[:,3] = outpts[:,3] - outptsI[:,3]
		
		x0,y0,z0 = outpts[0,:3]
		x = np.sqrt((outpts[:,0]-x0)**2+(outpts[:,1]-y0)**2+(outpts[:,2]-z0)**2)
		y = outpts[:,3]
		if verticalPlot:
			temp = x; x = y; y = temp
			temp = xlabel; xlabel = ylabel; ylabel = temp
			temp = xlims; xlims = ylims; ylims = temp
		if elevationPlot:
			x = outpts[:,3]
			y = outpts[:,2]
			temp = xlabel; xlabel = ylabel; ylabel = temp
			temp = xlims; xlims = ylims; ylims = temp
		plt.plot(x,y,marker,color=color)
		
		if xlims: ax.set_xlim(xlims)
		if ylims: ax.set_ylim(ylims)
		if xlabel: plt.xlabel(xlabel,size=font_size)		
		if ylabel: plt.ylabel(ylabel,size=font_size)
		if title: plt.title(title,size=font_size)
		for t in ax.get_xticklabels():
			t.set_fontsize(font_size)
		for t in ax.get_yticklabels():
			t.set_fontsize(font_size)
		
		extension, save_fname, pdf = save_name(save,variable=variable,time=time)
		plt.savefig(save_fname, dpi=100, facecolor='w', edgecolor='w',orientation='portrait', 
		format=extension,transparent=True, bbox_inches=None, pad_inches=0.1)
		if pdf: 
			os.system('epstopdf ' + save_fname)
			os.remove(save_fname)	
	def cutaway_plot(self,variable=None,time=None,divisions=[20,20,20],levels=10,cbar=False,angle=[45,45],xlims=[],method='nearest',
		ylims=[],zlims=[],colors='k',linestyle='-',save='',	xlabel='x / m', ylabel='y / m', zlabel='z / m', title='', 
		font_size='medium',equal_axes=True,grid_lines=None):
		'''Returns a filled plot of contour data on each of 3 planes in a cutaway plot. Invokes the ``slice()`` method to interpolate slice data for plotting.
		
		:param variable: Output data variable, for example 'P' = pressure.
		:type variable: str
		:param time: Time for which output data is requested. Can be supplied via ``fcontour.times`` list. Default is most recently available data. If a list of two times is passed, the difference between the first and last is plotted.
		:type time: fl64
		:param divisions: Resolution to supply mesh data in [x,y,z] coordinates.
		:type divisions: [int,int,int]
		:param levels: Contour levels to plot. Can specify specific levels in list form, or a single integer indicating automatic assignment of levels. 
		:type levels: lst[fl64], int
		:param cbar: Include colour bar.
		:type cbar: bool
		:param angle: 	View angle of zone. First number is tilt angle in degrees, second number is azimuth. Alternatively, if angle is 'x', 'y', 'z', view is aligned along the corresponding axis.
		:type angle: [fl64,fl64], str
		:param method: Method of interpolation, options are 'nearest', 'linear'.
		:type method: str
		:param xlims: Plot limits on x-axis.
		:type xlims: [fl64, fl64]
		:param ylims: Plot limits on y-axis.
		:type ylims: [fl64, fl64]
		:param zlims: Plot limits on z-axis.
		:type zlims: [fl64, fl64]
		:param colors: Specify colour string for contour levels.
		:type colors: lst[str]
		:param linestyle: Style of contour lines, e.g., 'k-' = solid black line, 'r:' red dotted line.
		:type linestyle: str
		:param save: Name to save plot. Format specified extension (default .png if none give). Supported extensions: .png, .eps, .pdf.
		:type save: str
		:param xlabel: Label on x-axis.
		:type xlabel: str
		:param ylabel: Label on y-axis.
		:type ylabel: str
		:param zlabel: Label on z-axis.
		:type zlabel: str
		:param title: Plot title.
		:type title: str
		:param font_size: Specify text size, either as an integer or string, e.g., 10, 'small', 'x-large'.
		:type font_size: str, int
		:param equal_axes: Force plotting with equal aspect ratios for all axes.
		:type equal_axes: bool
		:param grid_lines: Extend tick lines across plot according to specified linestyle, e.g., 'k:' is a dotted black line.
		:type grid_lines: bool
		 
		'''	
		save = os_path(save)
		# check inputs
		if time==None: time = self.times[-1]
		delta = False
		if isinstance(time,list) or isinstance(time,np.ndarray):
			if len(time)>1: 
				time0 = np.min(time)
				time = np.max(time)
				delta=True
		return_flag = self._check_inputs(variable=variable,time=time,slice=slice)
		if return_flag: return
		# set up axes
		fig = plt.figure(figsize=[11.7,8.275])
		ax = plt.axes(projection='3d')
		ax.set_aspect('equal', 'datalim')
		# make axes equal
		if 'x' not in self.variables or 'y' not in self.variables or 'z' not in self.variables: 
			pyfehm_print('ERROR: No xyz data, skipping',self._silent) 
			return
		xmin,xmax = np.min(self[time]['x']),np.max(self[time]['x'])
		ymin,ymax = np.min(self[time]['y']),np.max(self[time]['y'])
		zmin,zmax = np.min(self[time]['z']),np.max(self[time]['z'])		
		if equal_axes:
			MAX = np.max([xmax-xmin,ymax-ymin,zmax-zmin])/2
			C = np.array([xmin+xmax,ymin+ymax,zmin+zmax])/2
			for direction in (-1, 1):
				for point in np.diag(direction * MAX * np.array([1,1,1])):
					ax.plot([point[0]+C[0]], [point[1]+C[1]], [point[2]+C[2]], 'w')
		if not xlims: xlims = [xmin,xmax]
		if not ylims: ylims = [ymin,ymax]
		if not zlims: zlims = [zmin,zmax]
		# set view angle
		ax.view_init(angle[0],angle[1])
		
		ax.set_xlabel(xlabel,size=font_size)
		ax.set_ylabel(ylabel,size=font_size)
		ax.set_zlabel(zlabel,size=font_size)

		plt.title(title+'\n\n\n\n',size=font_size)
		
		scale = 1e6
		levels = [l/scale for l in levels]
	
		X, Y, Z, valsI = self.slice(variable, [[xlims[0],ylims[0],zlims[0]],[xlims[1],ylims[1],zlims[0]]], [divisions[0],divisions[1]], time, method)
		if delta:
			X, Y, Z, valsIi = self.slice(variable, [[xlims[0],ylims[0],zlims[0]],[xlims[1],ylims[1],zlims[0]]], [divisions[0],divisions[1]], time0, method)
			valsI = valsI - valsIi
		cset = ax.contourf(X, Y, valsI/scale, zdir='z', offset=zlims[0], cmap=cm.coolwarm,levels=levels)
		
		X, Y, Z, valsI = self.slice(variable, [[xlims[0],ylims[0],zlims[0]],[xlims[0],ylims[1],zlims[1]]], [divisions[1],divisions[2]], time, method)
		if delta:
			X, Y, Z, valsIi = self.slice(variable, [[xlims[0],ylims[0],zlims[0]],[xlims[0],ylims[1],zlims[1]]], [divisions[1],divisions[2]], time0, method)
			valsI = valsI - valsIi
		cset = ax.contourf(valsI/scale, Y, Z,  zdir='x', offset=xlims[0], cmap=cm.coolwarm,levels=levels)
		
		X, Y, Z, valsI = self.slice(variable, [[xlims[0],ylims[0],zlims[0]],[xlims[1],ylims[0],zlims[1]]], [divisions[0],divisions[2]], time, method)
		if delta:
			X, Y, Z, valsIi = self.slice(variable, [[xlims[0],ylims[0],zlims[0]],[xlims[1],ylims[0],zlims[1]]], [divisions[0],divisions[2]], time0, method)
			valsI = valsI - valsIi
		cset = ax.contourf(X, valsI/scale, Z,  zdir='y', offset=ylims[0], cmap=cm.coolwarm,levels=levels)

		if cbar:
			cbar=plt.colorbar(cset)
			tick_labels = [str(float(t*scale)) for t in levels]
			cbar.locator = matplotlib.ticker.FixedLocator(levels)
			cbar.formatter = matplotlib.ticker.FixedFormatter(tick_labels)
			cbar.update_ticks()

		if grid_lines:
			# add grid lines
			ax.set_xlim(xlims[0],xlims[1])
			ax.set_ylim(ylims[0],ylims[1])
			ax.set_zlim(zlims[0],zlims[1])
			xticks = ax.get_xticks()
			yticks = ax.get_yticks()
			zticks = ax.get_zticks()

			off = 0.
			for t in xticks:
				ax.plot([t,t],[ylims[0],ylims[1]],[zlims[0]+off,zlims[0]+off],grid_lines,zorder=100)
				ax.plot([t,t],[ylims[0]+off,ylims[0]+off],[zlims[0],zlims[1]],grid_lines,zorder=100)
			for t in yticks:
				ax.plot([xlims[0],xlims[1]],[t,t],[zlims[0]+off,zlims[0]+off],grid_lines,zorder=100)
				ax.plot([xlims[0]+off,xlims[0]+off],[t,t],[zlims[0],zlims[1]],grid_lines,zorder=100)
			for t in zticks:
				ax.plot([xlims[0],xlims[1]],[ylims[0]+off,ylims[0]+off],[t,t],grid_lines,zorder=100)
				ax.plot([xlims[0]+off,xlims[0]+off],[ylims[0],ylims[1]],[t,t],grid_lines,zorder=100)
		
		for t in ax.get_yticklabels():
			t.set_fontsize(font_size)	
		for t in ax.get_xticklabels():
			t.set_fontsize(font_size)	
		for t in ax.get_zticklabels():
			t.set_fontsize(font_size)	
		
		ax.set_xlim(xlims)
		ax.set_ylim(ylims)
		ax.set_zlim(zlims)
		
		extension, save_fname, pdf = save_name(save=save,variable=variable,time=time)
		plt.savefig(save_fname, dpi=100, facecolor='w', edgecolor='w',orientation='portrait', 
		format=extension,transparent=True, bbox_inches=None, pad_inches=0.1)
	def node(self,node,time=None,variable=None):
		'''Returns all information for a specific node.
		
		If time and variable not specified, a dictionary of time series is returned with variables as the dictionary keys.
		
		If only time is specified, a dictionary of variable values at that time is returned, with variables as dictionary keys.
		
		If only variable is specified, a time series vector is returned for that variable.
		
		If both time and variable are specified, a single value is returned, corresponding to the variable value at that time, at that node.
		
		:param node: Node index for which variable information required.
		:type node: int 
		:param time: Time at which variable information required. If not specified, all output.
		:type time: fl64
		:param variable: Variable for which information requested. If not specified, all output.
		:type variable: str
		
		'''
		if 'n' not in self.variables: 
			pyfehm_print('Node information not available',self._silent)
			return
		nd = np.where(self[self.times[0]]['n']==node)[0][0]
		if time is None and variable is None:
			ks = copy(self.variables); ks.remove('n')
			outdat = dict([(k,[]) for k in ks])
			for t in self.times:
				dat = self[t]
				for k in list(outdat.keys()):
					outdat[k].append(dat[k][nd])
		elif time is None:
			if variable not in self.variables: 
				pyfehm_print('ERROR: no variable by that name',self._silent)
				return
			outdat = []
			for t in self.times:
				dat = self[t]
				outdat.append(dat[variable][nd])
			outdat = np.array(outdat)
		elif variable is None:
			ks = copy(self.variables); ks.remove('n')
			outdat = dict([(k,self[time][k][nd]) for k in ks])			
		else:
			outdat = self[time][variable][nd]
		return outdat
	def paraview(self, grid, stor = None, exe = dflt.paraview_path,filename = 'temp.vtk',show=None,diff = False,zscale = 1., time_derivatives = False):
		""" Launches an instance of Paraview that displays the contour object.
		
		:param grid: Path to grid file associated with FEHM simulation that produced the contour output.
		:type grid: str
		:param stor: Path to grid file associated with FEHM simulation that produced the contour output.
		:type stor: str
		:param exe: Path to Paraview executable.
		:type exe: str
		:param filename: Name of VTK file to be output.
		:type filename: str
		:param show: Variable to show when Paraview starts up (default = first available variable in contour object).
		:type show: str
		:param diff: Flag to request PyFEHM to also plot differences of contour variables (from initial state) with time.
		:type diff: bool
		:param zscale: Factor by which to scale z-axis. Useful for visualising laterally extensive flow systems.
		:type zscale: fl64
		:param time_derivatives: Calculate new fields for time derivatives of contour data. For precision reasons, derivatives are calculated with units of 'per day'.
		:type time_derivatives: bool
		"""
		from fdata import fdata
		dat = fdata()
		dat.grid.read(grid,storfilename=stor)
		if show is None: 
			for var in self.variables:
				if var not in ['x','y','z','n']: break
			show = var
		dat.paraview(exe=exe,filename=filename,contour=self,show=show,diff=diff,zscale=zscale,time_derivatives=time_derivatives)
	def _get_variables(self): return self._variables
	variables = property(_get_variables)#: (*lst[str]*) List of variables for which output data are available.
	def _get_user_variables(self): return self._user_variables
	user_variables = property(_get_user_variables) #: (*lst[str]*) List of user-defined variables for which output data are available.
	def _get_format(self): return self._format
	format = property(_get_format) #: (*str*) Format of output file, options are 'tec', 'surf', 'avs' and 'avsx'.
	def _get_filename(self): return self._filename
	filename = property(_get_filename)  #: (*str*) Name of FEHM contour output file. Wildcards can be used to define multiple input files.
	def _get_times(self): return np.sort(self._times)
	times = property(_get_times)	#: (*lst[fl64]*) List of times (in seconds) for which output data are available.
	def _get_material_properties(self): return self._material_properties
	def _set_material_properties(self,value): self._material_properties = value
	material_properties = property(_get_material_properties, _set_material_properties) #: (*lst[str]*) List of material properties, keys for the material attribute.
	def _get_material(self): return self._material
	def _set_material(self,value): self._material = value
	material = property(_get_material, _set_material) #: (*dict[str]*) Dictionary of material properties, keyed by property name, items indexed by node_number - 1. This attribute is empty if no material property file supplied.
	def _get_x(self): return self._x
	def _set_x(self,value): self._x = value
	x = property(_get_x, _set_x) #: (*lst[fl64]*) Unique list of nodal x-coordinates for grid.
	def _get_y(self): return self._y
	def _set_y(self,value): self._y = value
	y = property(_get_y, _set_y) #: (*lst[fl64]*) Unique list of nodal y-coordinates for grid.
	def _get_z(self): return self._z
	def _set_z(self,value): self._z = value
	z = property(_get_z, _set_z) #: (*lst[fl64]*) Unique list of nodal z-coordinates for grid.
	def _get_xmin(self): return self._xmin
	def _set_xmin(self,value): self._xmin = value
	xmin = property(_get_xmin, _set_xmin) #: (*fl64*) Minimum nodal x-coordinate for grid.
	def _get_xmax(self): return self._xmax
	def _set_xmax(self,value): self._xmax = value
	xmax = property(_get_xmax, _set_xmax) #: (*fl64*) Maximum nodal x-coordinate for grid.
	def _get_ymin(self): return self._ymin
	def _set_ymin(self,value): self._ymin = value
	ymin = property(_get_ymin, _set_ymin) #: (*fl64*) Minimum nodal y-coordinate for grid.
	def _get_ymax(self): return self._ymax
	def _set_ymax(self,value): self._ymax = value
	ymax = property(_get_ymax, _set_ymax) #: (*fl64*) Maximum nodal y-coordinate for grid.
	def _get_zmin(self): return self._zmin
	def _set_zmin(self,value): self._zmin = value
	zmin = property(_get_zmin, _set_zmin) #: (*fl64*) Minimum nodal z-coordinate for grid.
	def _get_zmax(self): return self._zmax
	def _set_zmax(self,value): self._zmax = value
	zmax = property(_get_zmax, _set_zmax) #: (*fl64*) Maximum nodal z-coordinate for grid.
	def _get_information(self):
		print('FEHM contour output - format '+self._format)
		print('	call format: fcontour[time][variable][node_index-1]')
		prntStr =  '	times ('+str(len(self.times))+'): '
		for time in self.times: prntStr += str(time)+', '
		print(prntStr[:-2]+' days')
		prntStr = '	variables: '
		for var in self.variables: prntStr += str(var)+', '
		for var in self.user_variables: prntStr += str(var)+', '
		print(prntStr)
	what = property(_get_information) #:(*str*) Print out information about the fcontour object.
class fhistory(object):						# Reading and plotting methods associated with history output data.
	'''History output information object.
	
	'''
	def __init__(self,filename=None,verbose=True):
		self._filename=None	
		self._silent = dflt.silent
		self._format = ''
		self._times=[]	
		self._verbose = verbose
		self._data={}
		self._row=None
		self._nodes=[]	
		self._zones = []
		self._variables=[] 
		self._user_variables = []
		self._keyrows={}
		self.column_name=[]
		self.num_columns=0
		self._nkeys=1
		filename = os_path(filename)
		if filename: self._filename=filename; self.read(filename)
	def __getitem__(self,key):
		if key in self.variables or key in self.user_variables:
			return self._data[key]
		else: return None
	def __repr__(self): 
		retStr =  'History output for variables '
		for var in self.variables:
			retStr += var+', '
		retStr = retStr[:-2] + ' at '
		if len(self.nodes)>10:
			retStr += str(len(self.nodes)) + ' nodes.'
		else:
			if len(self.nodes)==1:
				retStr += 'node '
			else:
				retStr += 'nodes '
			for nd in self.nodes:
				retStr += str(nd) + ', '
			retStr = retStr[:-2] + '.'
		return retStr
	def read(self,filename): 						# read contents of file
		'''Read in FEHM history output information.
		
		:param filename: File name for output data, can include wildcards to define multiple output files.
		:type filename: str
		'''
		from glob import glob
		import re
		glob_pattern = re.sub(r'\[','[[]',filename)
		glob_pattern = re.sub(r'(?<!\[)\]','[]]', glob_pattern)
		files=glob(glob_pattern)
		configured=False
		for i,fname in enumerate(files):
			if self._verbose:
				pyfehm_print(fname,self._silent)
			self._file=open(fname,'rU')
			header=self._file.readline()
			if header.strip()=='': continue				# empty file
			self._detect_format(header)
			if self.format=='tec': 
				header=self._file.readline()
				if header.strip()=='': continue 		# empty file
				i = 0; sum_file = False
				while not header.startswith('variables'): 
					header=self._file.readline()
					i = i+1
					if i==10: sum_file=True; break
				if sum_file: continue
				self._setup_headers_tec(header)
			elif self.format=='surf': 
				self._setup_headers_surf(header)
			elif self.format=='default': 
				header=self._file.readline()
				header=self._file.readline()
				if header.strip()=='': continue 		# empty file
				i = 0; sum_file = False
				while not header.startswith('Time '): 
					header=self._file.readline()
					i = i+1
					if i==10: sum_file=True; break
				if sum_file: continue
				self._setup_headers_default(header)
			else: pyfehm_print('Unrecognised format',self._silent);return
			if not configured:
				self.num_columns = len(self.nodes)+1
			if self.num_columns>0: configured=True
			if self.format=='tec':
				self._read_data_tec(fname.split('_')[-2])
			elif self.format=='surf':
				self._read_data_surf(fname.split('_')[-2])
			elif self.format=='default':
				self._read_data_default(fname.split('_')[-1].split('.')[0])
			self._file.close()
	def _detect_format(self,header):
		if header.startswith('TITLE'):
			self._format = 'tec'
		elif header.startswith('Time '):
			self._format = 'surf'
		else:
			self._format = 'default'
	def _setup_headers_tec(self,header):
		header=header.split('" "Node')
		if self.nodes: return
		for key in header[1:-1]: self._nodes.append(int(key))
		self._nodes.append(int(header[-1].split('"')[0]))
	def _setup_headers_surf(self,header):
		header=header.split(', Node')
		if self.nodes: return
		for key in header[1:]: self._nodes.append(int(key))
	def _setup_headers_default(self,header):
		header=header.split(' Node')
		if self.nodes: return
		for key in header[1:]: self._nodes.append(int(key))
	def _read_data_tec(self,var_key):
		self._variables.append(hist_var_names[var_key])
		lns = self._file.readlines()
		i = 0
		while lns[i].startswith('text'): i+=1
		data = []
		for ln in lns[i:]: data.append([float(d) for d in ln.strip().split()])
		data = np.array(data)
		if data[-1,0]<data[-2,0]: data = data[:-1,:]
		self._times = np.array(data[:,0])
		self._data[hist_var_names[var_key]] = dict([(node,data[:,icol+1]) for icol,node in enumerate(self.nodes)])
	def _read_data_surf(self,var_key):
		self._variables.append(hist_var_names[var_key])
		lns = self._file.readlines()
		data = []
		for ln in lns: data.append([float(d) for d in ln.strip().split(',')])
		data = np.array(data)
		if data[-1,0]<data[-2,0]: data = data[:-1,:]
		self._times = np.array(data[:,0])
		self._data[hist_var_names[var_key]] = dict([(node,data[:,icol+1]) for icol,node in enumerate(self.nodes)])
	def _read_data_default(self,var_key):
		self._variables.append(hist_var_names[var_key])
		lns = self._file.readlines()
		data = []
		for ln in lns: data.append([float(d) for d in ln.strip().split()])
		data = np.array(data)
		if data[-1,0]<data[-2,0]: data = data[:-1,:]
		self._times = np.array(data[:,0])
		self._data[hist_var_names[var_key]] = dict([(node,data[:,icol+1]) for icol,node in enumerate(self.nodes)])
	def time_plot(self, variable=None, node=0, t_lim=[],var_lim=[],marker='x-',color='k',save='',xlabel='',ylabel='',
		title='',font_size='medium',scale=1.,scale_t=1.): 		# produce a time plot
		'''Generate and save a time series plot of the history data.
		
		:param variable: Variable to plot.
		:type variable: str
		:param node: Node number to plot.
		:type node: int
		:param t_lim: Time limits on x axis.
		:type t_lim: lst[fl64,fl64]
		:param var_lim: Variable limits on y axis.
		:type var_lim: lst[fl64,fl64]
		:param marker: String denoting marker and linetype, e.g., ':s', 'o--'. Default is 'x-' (solid line with crosses).
		:type marker: str
		:param color: String denoting colour. Default is 'k' (black).
		:type color: str
		:param save: Name to save plot.
		:type save: str
		:param xlabel: Label on x axis.
		:type xlabel: str
		:param ylabel: Label on y axis.
		:type ylabel: str
		:param title: Title of plot.
		:type title: str
		:param font_size: Font size for axis labels.
		:type font_size: str
		:param scale: If a single number is given, then the output variable will be multiplied by this number. If a two element list is supplied then the output variable will be transformed according to y = scale[0]*x+scale[1]. Useful for transforming between coordinate systems.
		:type scale: fl64
		:param scale_t: As for scale but applied to the time axis.
		:type scale_t: fl64
		'''
		save = os_path(save)
		if not node: pyfehm_print('ERROR: no plot node specified.',self._silent); return
		if not variable: 
			s = ['ERROR: no plot variable specified.']
			s.append('Options are')
			for var in self.variables: s.append(var)
			s = '\n'.join(s)
			pyfehm_print(s,self._silent)
			return True
		if not node: 
			s = ['ERROR: no plot node specified.']
			s.append('Options are')
			for node in self.nodes: s.append(node)
			s = '\n'.join(s)
			pyfehm_print(s,self._silent)
			return True
		
		plt.clf()
		plt.figure(figsize=[8,8])
		ax = plt.axes([0.15,0.15,0.75,0.75])
		if not isinstance(scale,list):
			if not isinstance(scale_t,list):
				plt.plot(self.times*scale_t,self[variable][node]*scale,marker)
			elif len(scale_t) == 2:
				plt.plot(self.times*scale_t[0]+scale_t[1],self[variable][node]*scale,marker)
		elif len(scale) == 2:
			if not isinstance(scale_t,list):
				plt.plot(self.times*scale_t,self[variable][node]*scale[0]+scale[1],marker)
			elif len(scale_t) == 2:
				plt.plot(self.times*scale_t[0]+scale_t[1],self[variable][node]*scale[0]+scale[1],marker)
		if t_lim: ax.set_xlim(t_lim)
		if var_lim: ax.set_ylim(var_lim)
		if xlabel: plt.xlabel(xlabel,size=font_size)		
		if ylabel: plt.ylabel(ylabel,size=font_size)
		if title: plt.title(title,size=font_size)
		for t in ax.get_xticklabels():
			t.set_fontsize(font_size)
		for t in ax.get_yticklabels():
			t.set_fontsize(font_size)
		
		extension, save_fname, pdf = save_name(save,variable=variable,node=node)
		plt.savefig(save_fname, dpi=100, facecolor='w', edgecolor='w',orientation='portrait', 
		format=extension,transparent=True, bbox_inches=None, pad_inches=0.1)
		if pdf: 
			os.system('epstopdf ' + save_fname)
			os.remove(save_fname)	
	def new_variable(self,name,node,data): 	
		'''Creates a new variable, which is some combination of the available variables.
		
		:param name: Name for the variable.
		:type name: str
		:param time: Node key which the variable should be associated with. Must be one of the existing keys, i.e., an item in fhistory.nodes.
		:type time: fl64
		:param data: Variable data, most likely some combination of the available parameters, e.g., pressure*temperature, pressure[t=10] - pressure[t=5]
		:type data: lst[fl64]
		'''
		if node not in self.nodes: 
			pyfehm_print('ERROR: supplied node must correspond to an existing node in fhistory.nodes',self._silent)
			return
		if name not in self._user_variables:
			self._data.update({name:dict([(nd,None) for nd in self.nodes])})
			if name not in self._user_variables:
				self._user_variables.append(name)
		self._data[name][node] = data
	def _get_variables(self): return self._variables
	variables = property(_get_variables)#: (*lst[str]*) List of variables for which output data are available.
	def _get_user_variables(self): return self._user_variables
	def _set_user_variables(self,value): self._user_variables = value
	user_variables = property(_get_user_variables, _set_user_variables) #: (*lst[str]*) List of user-defined variables for which output data are available.
	def _get_format(self): return self._format
	format = property(_get_format) #: (*str*) Format of output file, options are 'tec', 'surf', 'avs' and 'avsx'.
	def _get_filename(self): return self._filename
	filename = property(_get_filename)  #: (*str*) Name of FEHM contour output file. Wildcards can be used to define multiple input files.
	def _get_times(self): return np.sort(self._times)
	times = property(_get_times)	#: (*lst[fl64]*) List of times (in seconds) for which output data are available.
	def _get_nodes(self): return self._nodes
	nodes = property(_get_nodes)	#: (*lst[fl64]*) List of node indices for which output data are available.
	def _get_information(self):
		print('FEHM history output - format '+self._format)
		print('	call format: fhistory[variable][node][time_index]')
		prntStr = '	nodes: '
		for nd in self.nodes: prntStr += str(nd)+', '
		print(prntStr)
		prntStr =  '	times ('+str(len(self.times))+'): '
		for time in self.times: prntStr += str(time)+', '
		print(prntStr[:-2]+' days')
		prntStr = '	variables: '
		for var in self.variables: prntStr += str(var)+', '
		print(prntStr)
	what = property(_get_information) #:(*str*) Print out information about the fhistory object.
class fzoneflux(fhistory): 					# Derived class of fhistory, for zoneflux output
	'''Zone flux history output information object.
	'''
#	__slots__ = ['_filename','_times','_verbose','_data','_row','_zones','_variables','_keyrows','column_name','num_columns','_nkeys']
	def __init__(self,filename=None,verbose=True):
		super(fzoneflux,self).__init__(filename, verbose)
		self._filename=None	
		self._times=[]	
		self._verbose = verbose
		self._data={}
		self._row=None
		self._zones=[]	
		self._variables=[] 
		self._keyrows={}
		self.column_name=[]
		self.num_columns=0
		self._nkeys=1
		if filename: self._filename=filename; self.read(filename)
	def _setup_headers_tec(self,header):
		'placeholder'
	def _read_data_tec(self,var_key):
		zn = int(var_key[-5:])
		if var_key.startswith('c'):
			if zn not in self._zones: self._zones.append(zn)
			if 'co2_source' not in self._variables:
				self._variables += flxz_co2_names
				for var in flxz_co2_names: self._data[var] = {}
				
			lns = self._file.readlines()
			i = 0
			while lns[i].startswith('text'): i+=1
			data = []
			for ln in lns[i:]: data.append([float(d) for d in ln.strip().split()])
			data = np.array(data)
			if data[-1,0]<data[-2,0]: data = data[:-1,:]
			self._times = np.array(data[:,0])
			for j,var_key in enumerate(flxz_co2_names):
				self._data[var_key].update(dict([(zn,data[:,j+1])]))
		elif var_key.startswith('w'):
			if zn not in self._zones: self._zones.append(zn)
			if 'water_source' not in self._variables:
				self._variables += flxz_water_names
				for var in flxz_water_names: self._data[var] = {}
				
			lns = self._file.readlines()
			i = 0
			while lns[i].startswith('text'): i+=1
			data = []
			for ln in lns[i:]: data.append([float(d) for d in ln.strip().split()])
			data = np.array(data)			
			if data[-1,0]<data[-2,0]: data = data[:-1,:]
			self._times = np.array(data[:,0])
			for j,var_key in enumerate(flxz_water_names):
				self._data[var_key].update(dict([(zn,data[:,j+1])]))
		elif var_key.startswith('v'):
			if zn not in self._zones: self._zones.append(zn)
			if 'vapor_source' not in self._variables:
				self._variables += flxz_vapor_names
				for var in flxz_vapor_names: self._data[var] = {}
				
			lns = self._file.readlines()
			i = 0
			while lns[i].startswith('text'): i+=1
			data = []
			for ln in lns[i:]: data.append([float(d) for d in ln.strip().split()])
			data = np.array(data)			
			if data[-1,0]<data[-2,0]: data = data[:-1,:]
			self._times = np.array(data[:,0])
			for j,var_key in enumerate(flxz_vapor_names):
				self._data[var_key].update(dict([(zn,data[:,j+1])]))
	def _read_data_surf(self,var_key):
		self._variables.append(hist_var_names[var_key])
		lns = self._file.readlines()
		data = []
		for ln in lns[i:]: data.append([float(d) for d in ln.strip().split(',')])
		data = np.array(data)			
		if data[-1,0]<data[-2,0]: data = data[:-1,:]
		self._times = np.array(data[:,0])
		self._data[hist_var_names[var_key]] = dict([(node,data[:,icol+1]) for icol,node in enumerate(self.nodes)])
	def _read_data_default(self,var_key):
		self._variables.append(hist_var_names[var_key])
		lns = self._file.readlines()
		data = []
		for ln in lns[i:]: data.append([float(d) for d in ln.strip().split()])
		data = np.array(data)
		if data[-1,0]<data[-2,0]: data = data[:-1,:]
		self._times = np.array(data[:,0])
		self._data[hist_var_names[var_key]] = dict([(node,data[:,icol+1]) for icol,node in enumerate(self.nodes)])
	def _get_zones(self): return self._zones
	def _set_zones(self,value): self._zones = value
	zones = property(_get_zones, _set_zones) #: (*lst[int]*) List of zone indices for which output data are available.
class fnodeflux(object): 					# Reading and plotting methods associated with internode flux files.
	'''Internode flux information.
		
		Can read either water or CO2 internode flux files.
		
		The fnodeflux object is indexed first by node pair - represented as a tuple of node indices - and then
		by either the string 'liquid' or 'vapor'. Data values are in time order, as given in the 'times' attribute.
	'''
	def __init__(self,filename=None):
		self._filename = filename
		self._silent = dflt.silent
		self._nodepairs = []
		self._times = []
		self._timesteps = []
		self._data = {}
		if self._filename: self.read(self._filename)
	def __getitem__(self,key):
		if key in self.nodepairs:
			return self._data[key]
		else: return None
	def read(self,filename):
		'''Read in FEHM contour output information.
		
		:param filename: File name for output data, can include wildcards to define multiple output files.
		:type filename: str
		'''
		if not os.path.isfile(filename):
			pyfehm_print('ERROR: cannot find file at '+filename,self._silent)
			return
		fp = open(filename)
		lns = fp.readlines()
		N = int(lns[0].split()[1])
		
		data = np.zeros((N,len(lns)/(N+1),2)) 		# temporary data storage struc
		
		for ln in lns[1:N+1]:
			ln = ln.split()
			self._nodepairs.append((int(float(ln[0])),int(float(ln[1])))) 	# append node pair
		
		for i in range(len(lns)/(N+1)):
			ln = lns[(N+1)*i:(N+1)*(i+1)]
			
			nums = ln[0].split()
			self._timesteps.append(float(nums[2]))
			self._times.append(float(nums[3]))
			for j,lni in enumerate(ln[1:]):
				lnis = lni.split()
				data[j,i,0] = float(lnis[2])
				data[j,i,1] = float(lnis[3])
				
		for i,nodepair in enumerate(self.nodepairs):
			self._data[nodepair] = dict([(var,data[i,:,icol]) for icol,var in enumerate(['vapor','liquid'])])
			
	def _get_filename(self): return self._filename
	def _set_filename(self,value): self._filename = value
	filename = property(_get_filename, _set_filename) #: (*str*) filename target for internode flux file.
	def _get_timesteps(self): return np.sort(self._timesteps)
	def _set_timesteps(self,value): self._timesteps = value
	timesteps = property(_get_timesteps, _set_timesteps) #: (*lst*) timestep for which node flux information is reported.
	def _get_times(self): return np.sort(self._times)
	def _set_times(self,value): self._times = value
	times = property(_get_times, _set_times) #: (*lst*) times for which node flux information is reported.
	def _get_nodepairs(self): return self._nodepairs
	def _set_nodepairs(self,value): self._nodepairs = value
	nodepairs = property(_get_nodepairs, _set_nodepairs) #: (*lst*) node pairs for which node flux information is available. Each node pair is represented as a two item tuple of node indices.
class ftracer(fhistory): 					# Derived class of fhistory, for tracer output
	'''Tracer history output information object.
	'''
	def __init__(self,filename=None,output_filename=True):
		super(ftracer,self).__init__(filename, output_filename)
		self._filename=filename
		self._output_filename = output_filename	
		self._times=[]	
		self._data={}
		self._row=None
		self._nodes=[]	
		self._variables=[] 
		self._keyrows={}
		self.column_name=[]
		self.num_columns=0
		self._nkeys=1
		if filename: self.read(filename)
		if output_filename: self.read_output(output_filename)
	def read(self,filename):
		with open(filename,'r') as fp:
			lns = fp.readlines()
		Nnds = int(lns[2].strip())			# number of nodes
		for i in range(Nnds):				# indices of nodes
			self.nodes.append(int(lns[3+i].strip().split()[0]))
		Nsps = int(lns[3+Nnds].strip().split()[0])  # number of species

		Nt = int(len(lns[4+Nnds:])/(2*Nsps))   # number of timesteps
		ts = []
		Cs = [[] for j in range(Nsps)]
		for i in range(Nt):
			ts.append(float(lns[2*Nsps*i+4+Nnds].split()[0].strip()))
			for j in range(Nsps):
				Cs[j].append([float(v) for v in lns[2*Nsps*i+2*j+4+Nnds+1].strip().split()])

		for j in range(Nsps):
			v = 'Caq{:03d}'.format(j+1)
			self._variables.append(v)
			C = np.array(Cs[j])
			self._data[v] = dict([(node,C[:,icol]) for icol,node in enumerate(self.nodes)])
		self._times = np.array(ts)
	def read_output(self, output_filename):
		with open(output_filename,'r') as fp:
			lns = fp.readlines()
		Nsps = len(self._variables)
		Cs = [[] for j in range(Nsps)]
		qs = []
		ts = []
		dts = []
		for i,ln in enumerate(lns):
			if ln.strip().startswith('Timing Information'):
				t = float(lns[i+2].split()[1])
				dt = float(lns[i+2].split()[2])*3600*24
				
			if ln.strip().startswith('Nodal Information (Water)'):
				q = [float(lns[i+3+j].split()[5]) for j in range(len(self.nodes))]
				
			if ln.strip().startswith('Solute output information'):
				ts.append(t)
				dts.append(dt)
				qs.append(q)
				ind = int(ln.strip().split()[-1])
				Cs[ind-1].append([float(lns[i+3+j].split()[4]) for j in range(len(self.nodes))])

		qs = np.array(qs)
		for j in range(Nsps):
			v = 'Caq{:03d}_src'.format(j+1)
			self._variables.append(v)
			C = np.array(Cs[j])
			self._data[v] = dict([(node,np.interp(self.times, ts, C[:,icol].T/qs[:,icol])) for icol,node in enumerate(self.nodes)])
		
class fptrk(fhistory): 						# Derived class of fhistory, for particle tracking output
	'''Tracer history output information object.
	'''
	def __init__(self,filename=None,verbose=True):
		super(fptrk,self).__init__(filename, verbose)
		self._filename=None	
		self._silent = dflt.silent
		self._times=[]	
		self._verbose = verbose
		self._data={}
		self._row=None
		self._nodes=[0]	
		self._variables=[] 
		self._keyrows={}
		self.column_name=[]
		self.num_columns=0
		self._nkeys=1
		if filename: self._filename=filename; self.read(filename)
	def read(self,filename): 						# read contents of file
		'''Read in FEHM particle tracking output information. Index by variable name.
		
		:param filename: File name for output data, can include wildcards to define multiple output files.
		:type filename: str
		'''
		from glob import glob
		import re
		glob_pattern = re.sub(r'\[','[[]',filename)
		glob_pattern = re.sub(r'(?<!\[)\]','[]]', glob_pattern)
		files=glob(glob_pattern)
		configured=False
		for i,fname in enumerate(files):
			if self._verbose:
				pyfehm_print(fname,self._silent)
			self._file=open(fname,'rU')
			header=self._file.readline()
			if header.strip()=='': continue				# empty file
			header=self._file.readline()
			self._setup_headers_default(header)
			self._read_data_default()
			self._file.close()
	def _setup_headers_default(self,header):
		header=header.strip().split('"')[3:-1]
		header = [h for h in header if h != ' ']
		for var in header: self._variables.append(var)
	def _read_data_default(self):
		lns = self._file.readlines()
		data = []
		for ln in lns: data.append([float(d) for d in ln.strip().split()])
		data = np.array(data)
		if data[-1,0]<data[-2,0]: data = data[:-1,:]
		self._times = np.array(data[:,0])
		self._data = dict([(var,data[:,icol+1]) for icol,var in enumerate(self.variables)])

class multi_pdf(object):
	'''Tool for making a single pdf document from multiple eps files.'''
	def __init__(self,combineString = 'gswin64',
		save='multi_plot.pdf',files = [],delete_files = True):
		self.combineString = combineString
		self._silent = dflt.silent
		self._save = os_path(save)
		self._delete_files = delete_files
		self._assign_files(files)
	def _assign_files(self,files):
		if files == []: self._files = {}
		if isinstance(files,list):
			self._files = dict([(i+1,file) for i,file in enumerate(files)])
		elif isinstance(files,dict):
			ks = list(files.keys())
			for k in ks:
				if not isinstance(k,int):pyfehm_print('ERROR: Dictionary keys must be integers.',self._silent);return
			self._files = files
		elif isinstance(files,str):
			self._files = dict(((1,files),))
	def add(self,filename,pagenum=None):
		'''Add a new page. If a page number is specified, the page will replace the current. 
		Otherwise it will be appended to the end of the document.
		
		:param filename: Name of .eps file to be added.
		:type filename: str
		:param pagenum: Page number of file to be added.
		:type pagenum: int
		
		'''
		if len(filename.split('.'))==1: filename += '.eps'
		if not os.path.isfile(filename): print('WARNING: '+filename+' not found.'); return		
		if not filename.endswith('.eps'): print('WARNING: Non EPS format not supported.')
		
		if pagenum and pagenum in list(self.files.keys()):
			print('WARNING: Replacing '+self.files[pagenum])
			self.files[pagenum] = filename
		else: 
			if not pagenum: pagenum = self._pagemax+1
			self._files.update(dict(((pagenum,filename),)))			
	def insert(self,filename,pagenum):
		'''Insert a new page at the given page number.
		
		:param filename: Name of .eps file to be inserted.
		:type filename: str
		:param pagenum: Page number of file to be inserted.
		:type pagenum: int
		'''
		if len(filename.split('.'))==1: filename += '.eps'
		if not os.path.isfile(filename): print('WARNING: '+filename+' found.'); return
		if not filename.endswith('.eps'): print('WARNING: Non EPS format not supported.')
			
		if pagenum > self._pagemax: self.add(filename); return
		ks = list(self._files.keys())
		self._files = dict([(k,self._files[k]) for k in ks if k < pagenum]+
		[(pagenum,filename)]+[(k+1,self._files[k]) for k in ks if k >= pagenum])
	def make(self):
		'''Construct the pdf.'''
		cs = self.combineString + ' -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -sOutputFile='+self.save
		for i in np.sort(list(self.files.keys())):
			if not self.files[i].endswith('.eps'): print('WARNING: Cannot combine '+self.files[i]+'. EPS format required. Skipping...'); continue
			if len(self.files[i].split()) != 1:
				cs += ' "'+self.files[i]+'"'
			else:
				cs += ' '+self.files[i]
		os.system(cs)
		for i in np.sort(list(self.files.keys())): 
			if len(self.files[i].split()) != 1:
				os.system(delStr+' "'+self.files[i]+'"')
			else:
				os.system(delStr+' '+self.files[i])
	def _get_combineString(self): return self._combineString
	def _set_combineString(self,value): self._combineString = value
	combineString = property(_get_combineString, _set_combineString) #: (*str*)	Command line command, with options, generate pdf from multiple eps files. See manual for further instructions.
	def _get_files(self): return self._files
	files = property(_get_files) #: (*lst[str]*) List of eps files to be assembled into pdf.
	def _get_pagemax(self): 
		ks = list(self._files.keys())
		for k in ks:
			if not isinstance(k,int): print('ERROR: Non integer dictionary key'); return
		if len(ks) == 0: return 0
		return np.max(ks)
	_pagemax = property(_get_pagemax)
	def _get_save(self): return self._save
	def _set_save(self,value): self._save = value
	save = property(_get_save, _set_save) #: (*str*) Name of the final pdf to output.

"""Classes for VTK output."""
class fStructuredGrid(pv.StructuredGrid):
	def __init__(self,dimensions,points):
		pv.StructuredGrid.__init__(self,dimensions,points)
	def to_string(self, time = None, format='ascii'):
		t = self.get_datatype(self.points)
		ret = ['DATASET STRUCTURED_GRID']
		
		# include time information
		if time is not None:
			ret.append('FIELD FieldData 2')
			ret.append('TIME 1 1 double')
			ret.append('%8.7f'%time)
			ret.append('CYCLE 1 1 int')
			ret.append('123')
		
		ret.append('DIMENSIONS %s %s %s'%self.dimensions)
		ret.append('POINTS %s %s'%(self.get_size(),t))
		ret.append(self.seq_to_string(self.points,format,t))
		
		return '\n'.join(ret)
class fUnstructuredGrid(pv.UnstructuredGrid):
	def __init__(self,points,vertex=[],poly_vertex=[],line=[],poly_line=[],
				 triangle=[],triangle_strip=[],polygon=[],pixel=[],
				 quad=[],tetra=[],voxel=[],hexahedron=[],wedge=[],pyramid=[]):
		pv.UnstructuredGrid.__init__(self,points,vertex=vertex,poly_vertex=poly_vertex,line=line,poly_line=poly_line,
				 triangle=triangle,triangle_strip=triangle_strip,polygon=polygon,pixel=pixel,
				 quad=quad,tetra=tetra,voxel=voxel,hexahedron=hexahedron,wedge=wedge,pyramid=pyramid)			
	def to_string(self,time = None,format='ascii'):
		t = self.get_datatype(self.points)
		ret = ['DATASET UNSTRUCTURED_GRID']
		
		# include time information
		if time is not None:
			ret.append('FIELD FieldData 2')
			ret.append('TIME 1 1 double')
			ret.append('%8.7f'%time)
			ret.append('CYCLE 1 1 int')
			ret.append('123')
		
		ret.append('POINTS %s %s'%(self.get_size(),t))
		ret.append(self.seq_to_string(self.points,format,t))
		tps = []
		r = []
		sz = 0
		for k in list(self._vtk_cell_types_map.keys()):
			kv = getattr(self,k)
			if kv==[] or kv[0]==[]: continue
			s = self.seq_to_string([[len(v)]+list(v) for v in kv],format,'int')
			r .append(s)
			for v in kv:
				tps.append(self._vtk_cell_types_map[k])
				sz += len(v)+1
		sep = (format=='ascii' and '\n') or (format=='binary' and '')
		r = sep.join(r)
		ret += ['CELLS %s %s'%(len(tps),sz),
				r,
				'CELL_TYPES %s'%(len(tps)),
				self.seq_to_string(tps,format,'int')]
		return '\n'.join(ret)
class fVtkData(pv.VtkData):
	def __init__(self,*args,**kws):
		pv.VtkData.__init__(self,*args,**kws)
		self.times = []
		self.material = pv.PointData()
		self.contour = {}
	def to_string(self, time=None, format = 'ascii',material=False):
		ret = ['# vtk DataFile Version 2.0',
			   self.header,
			   format.upper(),
			   self.structure.to_string(time=time,format=format)
			   ]
		if self.cell_data.data:
			ret.append(self.cell_data.to_string(format=format))
		if material:
			ret.append(self.material.to_string(format=format))
		else:
			if self.contour[time].data:
				ret.append(self.contour[time].to_string(format=format))
		return '\n'.join(ret)
	def tofile(self, filename, format = 'ascii'):
		"""Save VTK data to file.
		"""
		written_files = []
		if not pv.common.is_string(filename):
			raise TypeError('argument filename must be string but got %s'%(type(filename)))
		if format not in ['ascii','binary']:
			raise TypeError('argument format must be ascii | binary')
		filename = filename.strip()
		if not filename:
			raise ValueError('filename must be non-empty string')
		if filename[-4:]!='.vtk':
			filename += '.vtk'
		
		# first write material properties file
		filename_int = ''.join(filename[:-4]+'_mat.vtk')
		f = open(filename_int,'wb')
		f.write(self.to_string(format=format,material=True).encode('utf-8'))
		f.close()
		written_files.append(filename_int)
		
		# write contour output file
		times = np.sort(list(self.contour.keys()))
		for i,time in enumerate(times):
			if len(times)>1:
				filename_int = ''.join(filename[:-4]+'.%04i'%i+'.vtk')
			else:
				filename_int = filename
			#print 'Creating file',`filename`
			f = open(filename_int,'wb')
			f.write(self.to_string(time=time,format=format).encode('utf-8'))
			f.close()
			written_files.append(filename_int)
		return written_files
	def tofilewell(self,filename,format = 'ascii'):
		"""Save VTK data to file.
		"""
		written_files = []
		if not pv.common.is_string(filename):
			raise TypeError('argument filename must be string but got %s'%(type(filename)))
		if format not in ['ascii','binary']:
			raise TypeError('argument format must be ascii | binary')
		filename = filename.strip()
		
		# first write material properties file
		f = open(filename,'wb')
		f.write(self.to_string(format=format,material=True).encode('utf-8'))
		f.close()
class fvtk(object):
	def __init__(self,parent,filename,contour,diff,zscale,spatial_derivatives,time_derivatives):
		self.parent = parent
		self.path = fpath(parent = self)
		self.path.filename = filename
		self.data = None
		self.csv = None
		self.contour = contour
		self.variables = []
		self.materials = []
		self.zones = []
		self.diff = diff
		self.spatial_derivatives = spatial_derivatives
		self.time_derivatives = time_derivatives
		self.zscale = zscale
		self.wells = None
	def __getstate__(self):
		return dict((k, getattr(self, k)) for k in self.__slots__)
	def __setstate__(self, data_dict):
		for (name, value) in data_dict.items():
			setattr(self, name, value)
	def assemble(self):		
		"""Assemble all information in pyvtk objects."""			
		self.assemble_grid()		# add grid information
		self.assemble_zones()		# add zone information
		self.assemble_properties()	# add permeability data
		if self.contour != None:	# add contour data
			self.assemble_contour()
	def assemble_grid(self):
		"""Assemble grid information in pyvtk objects."""
		# node positions, connectivity information
		nds = np.array([nd.position for nd in self.parent.grid.nodelist])
		if self.zscale != 1.: 
			zmin = np.min(nds[:,2])
			nds[:,2] = (nds[:,2]-zmin)*self.zscale+zmin
		if isinstance(self.parent.grid.elemlist[0],list):
			cns = [[nd-1 for nd in el] for el in self.parent.grid.elemlist]
		else:
			cns = [[nd.index-1 for nd in el.nodes] for el in self.parent.grid.elemlist]
		
		# make grid
		if len(cns[0]) == 3:
			self.data = fVtkData(fUnstructuredGrid(nds,triangle=cns),'PyFEHM VTK model output')
		elif len(cns[0]) == 4:
			self.data = fVtkData(fUnstructuredGrid(nds,tetra=cns),'PyFEHM VTK model output')
		elif len(cns[0]) == 8:
			self.data = fVtkData(fUnstructuredGrid(nds,hexahedron=cns),'PyFEHM VTK model output')
		else:
			print("ERROR: Number of connections in connectivity not recognized: "+str(len(cns[0])))
			return
		
		# grid information
		dat = np.array([nd.position for nd in self.parent.grid.nodelist])
		nds = np.array([nd.index for nd in self.parent.grid.nodelist])
		self.data.material.append(pv.Scalars(nds,name='n',lookup_table='default'))
		self.data.material.append(pv.Scalars(dat[:,0],name='x',lookup_table='default'))
		self.data.material.append(pv.Scalars(dat[:,1],name='y',lookup_table='default'))
		self.data.material.append(pv.Scalars(dat[:,2],name='z',lookup_table='default'))
		
		self.x_lim = [np.min(dat[:,0]),np.max(dat[:,0])]
		self.y_lim = [np.min(dat[:,1]),np.max(dat[:,1])]
		self.z_lim = [np.min(dat[:,2]),np.max(dat[:,2])]
		self.n_lim = [1,len(self.parent.grid.nodelist)]
	def assemble_zones(self):
		"""Assemble zone information in pyvtk objects."""
		# zones will be considered material properties as they only need to appear once
		N = len(self.parent.grid.nodelist)
		nds = np.zeros((1,N))[0]
		self.parent.zonelist.sort(key=lambda x: x.index)
		for zn in self.parent.zonelist:
			if zn.index == 0: continue
			name = 'zone%04i'%zn.index
			if zn.name: name += '_'+zn.name.replace(' ','_')
			self.zones.append(name)
			zn_nds = copy(nds)
			for nd in zn.nodelist: zn_nds[nd.index-1] = 1
			self.data.material.append(
				pv.Scalars(zn_nds,
				name=name,
				lookup_table='default'))
	def assemble_properties(self):
		"""Assemble material properties in pyvtk objects."""
		# permeabilities
		perms = np.array([nd.permeability for nd in self.parent.grid.nodelist])
		if not all(v is None for v in perms):
			if np.mean(perms)>0.: perms = np.log10(perms)
			self.add_material('perm_x',perms[:,0])
			self.add_material('perm_y',perms[:,1])
			self.add_material('perm_z',perms[:,2])
		else:
			blank = [-1.e30 for nd in self.parent.grid.nodelist]
			self.add_material('perm_x',blank)
			self.add_material('perm_y',blank)
			self.add_material('perm_z',blank)

		props = np.array([[nd.density, nd.porosity, nd.specific_heat, nd.youngs_modulus,nd.poissons_ratio,nd.thermal_expansion,nd.pressure_coupling] 	for nd in self.parent.grid.nodelist])
		names = ['density','porosity','specific_heat','youngs_modulus','poissons_ratio','thermal_expansion','pressure_coupling']
		for name, column in zip(names,props.T):
			self.add_material(name,column)
	def add_material(self,name,data):
		if all(v is None for v in data): return 		# if all None, no data to include
		data = np.array([dt if dt is not None else -1.e30 for dt in data]) 		# check for None, replace with -1.e30
		self.data.material.append(pv.Scalars(data,name=name,lookup_table='default'))
		self.materials.append(name)
		self.__setattr__(name+'_lim',[np.min(data),np.max(data)])
	def assemble_contour(self):
		"""Assemble contour output in pyvtk objects."""
		self.data.contour = dict([(time,pv.PointData()) for time in self.contour.times])
		if self.diff: time0 = self.contour.times[0]
		for time in self.contour.times:
			do_lims = (time == self.contour.times[-1])
			for var in self.contour.variables+self.contour.user_variables:
				# skip conditions
				if var in self.contour.variables:
					if time != self.contour.times[0] and var in ['x','y','z','n']: continue
				else:
					if var not in list(self.contour[time].keys()): continue
					if self.diff:
						if var not in list(self.contour[time0].keys()): continue
				
				# field for contour variable
				if var not in self.variables: self.variables.append(var)
				self.data.contour[time].append(pv.Scalars(self.contour[time][var],name=var,lookup_table='default'))
				if var in ['x','y','z','n']: continue
				if do_lims: self.__setattr__(var+'_lim',[np.min(self.contour[time][var]),np.max(self.contour[time][var])])
				
				# differences from initial value
				if self.diff:
					self.data.contour[time].append(pv.Scalars(self.contour[time][var]-self.contour[time0][var],name='diff_'+var,lookup_table='default'))
				# time derivatives
				if self.time_derivatives:
					# find position, determines type of differencing
					ind = np.where(time==self.contour.times)[0][0]
					if ind == 0: 
						# forward difference
						dt = self.contour.times[1]-time
						f0 = self.contour[time][var]
						f1 = self.contour[self.contour.times[1]][var]
						dat = (f1-f0)/dt
					elif ind == (len(self.contour.times)-1): 
						# backward difference
						dt = time-self.contour.times[-2]
						f0 = self.contour[self.contour.times[-2]][var]
						f1 = self.contour[time][var]
						dat = (f1-f0)/dt
					else:
						# central difference
						dt1 = time - self.contour.times[ind-1]
						dt2 = self.contour.times[ind+1] - time
						f0 = self.contour[self.contour.times[ind-1]][var]
						f1 = self.contour[time][var]
						f2 = self.contour[self.contour.times[ind+1]][var]
						dat = -dt2/(dt1*(dt1+dt2))*f0 + (dt2-dt1)/(dt1*dt2)*f1 + dt1/(dt2*(dt1+dt2))*f2
					self.data.contour[time].append(pv.Scalars(dat,name='d_'+var+'_dt',lookup_table='default'))
			if 'flux_x' in self.contour.variables and 'flux_y' in self.contour.variables and 'flux_z' in self.contour.variables:
				flux = [(self.contour[time]['flux_x'][i],self.contour[time]['flux_y'][i],self.contour[time]['flux_z'][i]) for i in range(len(self.contour[time]['flux_x']))]
				self.data.contour[time].append(pv.Vectors(flux,name='flux'))
			if 'flux_x_vap' in self.contour.variables and 'flux_y_vap' in self.contour.variables and 'flux_z_vap' in self.contour.variables:
				flux = [(self.contour[time]['flux_x_vap'][i],self.contour[time]['flux_y_vap'][i],self.contour[time]['flux_z_vap'][i]) for i in range(len(self.contour[time]['flux_x_vap']))]
				self.data.contour[time].append(pv.Vectors(flux,name='flux_vap'))

	def write(self):	
		"""Call to write out vtk files."""
		if self.parent.work_dir: wd = self.parent.work_dir
		else: wd = self.parent._path.absolute_to_file
		if wd is None: wd = ''
		else: wd += os.sep
		fls = self.data.tofile(wd+self.path.filename)
		# save file names for later use
		self.material_file = fls[0]
		self.contour_files = []
		if len(fls)>1:
			self.contour_files = fls[1:]
		return fls
	def write_wells(self,wells):
		"""Receives a dictionary of well track objects, creates the corresponding vtk grid.
		"""		
		nds = np.array([nd.position for nd in self.parent.grid.nodelist])
		zmin = np.min(nds[:,2])
		for k in list(wells.keys()):
			well = wells[k]
			if isinstance(well,np.ndarray):
				nds = well
				cns = [[i,i+1] for i in range(np.shape(nds)[0]-1)]
				grid = fVtkData(fUnstructuredGrid(nds,line=cns),'Well track: %s'%k)
				grid.material.append(pv.Scalars(np.ones((1,len(nds[:,2])))[0],name='T',lookup_table='default'))				
			else:
				nds = np.array([[well.location[0],well.location[1],(z-zmin)*self.zscale+zmin] for z in well.data[:,0]])
				cns = [[i,i+1] for i in range(np.shape(nds)[0]-1)]
				grid = fVtkData(fUnstructuredGrid(nds,line=cns),'RAGE well track: %s'%well.name)
				grid.material.append(pv.Scalars(well.data[:,1] ,name='T',lookup_table='default'))
			filename=k+'_wells.vtk'
			grid.tofilewell(filename)
		self.wells = list(wells.keys())
	def initial_display(self,show):
		"""Determines what variable should be initially displayed."""
		mat_vars = ['n','x','y','z','perm_x','perm_y','perm_z','porosity','density','cond_x','cond_y','cond_z']
		if self.contour:
			cont_vars = self.contour.variables
		# convert k* format to perm_*
		if show == 'kx': show = 'perm_x'
		elif show == 'ky': show = 'perm_y'
		elif show == 'kz': show = 'perm_z'
		
		# check for unspecified coordinate in potentially anisotropic properties
		if show in ['permeability','perm']:
			print('NOTE: plotting z-component of permeability, for other components specify show=\'perm_x\', etc.')
			show = 'perm_z'
		if show in ['conducitivity','cond']:
			print('NOTE: plotting z-component of conductivity, for other components specify show=\'cond_x\', etc.')
			show = 'cond_z'
		
		# check if material property or contour output requested for display
		if show in mat_vars:
			self.initial_show = 'material'
			self.default_material_property = show
			self.default_material_lims = self.__getattribute__(show+'_lim')
			if self.contour:
				# get default contour variable to display
				for var in self.contour.variables:
					if var not in ['x','y','z','n']: break
				self.default_contour_variable = var
				self.default_contour_lims = self.__getattribute__(var+'_lim')
		elif show in cont_vars:
			self.initial_show = 'contour'
			self.default_contour_variable = show
			self.default_contour_lims = self.__getattribute__(show+'_lim')
			self.default_material_property = 'perm_x' 		# default 
			self.default_material_lims = self.__getattribute__('perm_x_lim')
		else:
			print('ERROR: requested property or variable does not exist, available options are...')
			print('Material properties:')
			for mat in mat_vars:
				print('  - '+mat)
			print('Contour output variables:')
			for var in cont_vars:
				print('  - '+var)
			print('')
	def startup_script(self,nodes):
		x0,x1 = self.parent.grid.xmin, self.parent.grid.xmax
		y0,y1 = self.parent.grid.ymin, self.parent.grid.ymax
		z0,z1 = self.parent.grid.zmin, self.parent.grid.zmax
		z1 = self.zscale*(z1-z0)+z0
		xm,ym,zm = (x0+x1)/2., (y0+y1)/2., (z0+z1)/2.
		xr,yr,zr = (x1-x0), (y1-y0), (z1-z0)
		
		dflt_mat = '\''+self.default_material_property+'\''		
		mat_lim = self.default_material_lims
		
		f = open('pyfehm_paraview_startup.py','w')
		
		contour_files=[file for file in self.contour_files]
		
		################################### load paraview modules ######################################
		lns = [
			'try: paraview.simple',
			'except: from paraview.simple import *',
			'paraview.simple._DisableFirstRenderCameraReset()',
			'',
			]
			
		################################### load material properties ###################################
		lns += ['mat_prop = LegacyVTKReader( FileNames=[']
		file = self.material_file.replace('\\','/')
		lns += ['\''+file+'\',']
		lns += ['] )']
		lns += ['RenameSource("model", mat_prop)']
			
		################################### initial property display ###################################
		lns += [
			'rv = GetRenderView()',
			'dr = Show()',
			'dr.ScalarOpacityUnitDistance = 1.7320508075688779',
			'dr.EdgeColor = [0.0, 0.0, 0.5]',
			'',
			'rv.CenterOfRotation = [%10.5f, %10.5f, %10.5f]'%(xm,ym,zm),
			'',
			'rv.CameraViewUp = [-0.4, -0.11, 0.92]',
			'rv.CameraPosition = [%10.5f, %10.5f, %10.5f]'%(xm+2.5*xr,ym+1.5*yr,zm+1.5*zr),
			'rv.CameraFocalPoint = [%10.5f, %10.5f, %10.5f]'%(xm,ym,zm),
			'',
			'mr = GetDisplayProperties(mat_prop)',
			'mr.Representation = \'Surface With Edges\'',
			'',
			'lt = GetLookupTableForArray( '+dflt_mat+', 1, RGBPoints=[%4.2f, 0.23, 0.299, 0.754, %4.2f, 0.706, 0.016, 0.15], VectorMode=\'Magnitude\', NanColor=[0.25, 0.0, 0.0], ColorSpace=\'Diverging\', ScalarRangeInitialized=1.0 )'%tuple(mat_lim),
			'',
			'pf = CreatePiecewiseFunction( Points=[%4.2f, 0.0, 0.5, 0.0, %4.2f, 1.0, 0.5, 0.0] )'%tuple(mat_lim),
			'',
			'mr.ScalarOpacityFunction = pf',
			'mr.ColorArrayName = (\'POINT_DATA\', '+dflt_mat+')',
			'mr.LookupTable = lt',
			'',
			'lt.ScalarOpacityFunction = pf',
			'',
			'ScalarBarWidgetRepresentation1 = CreateScalarBar( Title='+dflt_mat+', LabelFontSize=12, Enabled=1, TitleFontSize=12 )',
			'GetRenderView().Representations.append(ScalarBarWidgetRepresentation1)',
			'',
			'lt = GetLookupTableForArray('+dflt_mat+', 1 )',
			'',
			'ScalarBarWidgetRepresentation1.LookupTable = lt',
			'',
			]
			
		################################### load in nodes as glyphs ###################################	
		ndRadius = np.min([con.distance for con in self.parent.grid.connlist])/10.
		ndRadius = ndRadius*self.zscale
		if nodes:
			lns += [
				'AnimationScene1 = GetAnimationScene()',
				'AnimationScene1.AnimationTime = 0.0',
				'rv.ViewTime = 0.0',
				'source = FindSource("model")',
				'SetActiveSource(source)',
				'',
				'G = Glyph( GlyphType="Arrow", GlyphTransform="Transform2" )',
				'G.GlyphTransform = "Transform2"',
				'G.GlyphType = "Sphere"',
				'G.RandomMode = 0',
				'G.ScaleMode = \'off\'',
				'G.MaskPoints = 0',
				'G.GlyphType.Radius = %10.5f'%ndRadius,
				'',
				'RenameSource("nodes", G)',
				'',
				'rv = GetRenderView()',
				'mr = GetDisplayProperties(source)',
				'dr = Show()',
				'dr.ColorArrayName = (\'POINT_DATA\', \'n\')',
				'dr.ScaleFactor = 1.1',
				'dr.SelectionPointFieldDataArrayName = "nodes"',
				'dr.EdgeColor = [0.0, 0.0, 0.5000076295109483]',
				'dr.ColorArrayName = (\'POINT_DATA\', \'\')',
				'dr.DiffuseColor = [0.,0.,0.]',
				'dr.Visibility = 0',
				]
		################################### load in zones as glyphs ###################################		
		colors = [
			[1.,1.,0.],
			[1.,0.,1.],
			[0.,1.,1.],
			[1.,1.,0.5],
			[1.,0.5,1.],
			[0.5,1.,1.],
			[1.,1.,0.25],
			[1.,0.25,1.],
			[0.25,1.,1.],
			[1.,1.,0.75],
			[1.,0.75,1.],
			[0.75,1.,1.],
			
			[0.5,1.,0.5],
			[1.,0.5,0.5],
			[0.5,0.5,1.],			
			[0.5,0.75,0.5],
			[0.75,0.5,0.5],
			[0.5,0.5,0.75],		
			[0.5,0.25,0.5],
			[0.25,0.5,0.5],
			[0.5,0.5,0.25],
			
			[0.75,1.,0.75],
			[1.,0.75,0.75],
			[0.75,0.75,1.],			
			[0.75,0.5,0.75],
			[0.5,0.75,0.75],
			[0.75,0.75,0.5],			
			[0.75,0.25,0.75],
			[0.25,0.75,0.75],
			[0.75,0.75,0.25],			
			
			[0.25,1.,0.25],
			[1.,0.25,0.25],
			[0.25,0.25,1.],			
			[0.25,0.75,0.25],
			[0.75,0.25,0.25],
			[0.25,0.25,0.75],			
			[0.25,0.5,0.25],
			[0.5,0.25,0.25],
			[0.25,0.25,0.5],
			]
		
		zones = []; cols = []
		for zone,color in zip(self.zones,colors):
			if self.show_zones == 'user':		
				if ('XMIN' in zone) or ('XMAX' in zone) or ('YMIN' in zone) or ('YMAX' in zone) or ('ZMIN' in zone) or ('ZMAX' in zone): continue
			elif self.show_zones == 'none': continue
			elif isinstance(self.show_zones,list):
				if zone not in ['zone%04i_%s'%(zn.index,zn.name) for zn in self.show_zones]: continue
			zones.append(zone)
			cols.append(color)
		
		lns += ['cols = [']
		for col in cols:
			lns += ['[%3.2f,%3.2f,%3.2f],'%tuple(col)]
		lns += [']']
		lns += ['zones = [']
		for zone in zones:
			lns += ['\''+zone+'\',']
		lns += [']']
		lns += ['for zone,col in zip(zones,cols):',
			'\tAnimationScene1 = GetAnimationScene()',
			'\tAnimationScene1.AnimationTime = 0.0',
			'\trv.ViewTime = 0.0',
			'\tsource = FindSource("model")',
			'\tSetActiveSource(source)',
			'\t',
			'\tG = Glyph( GlyphType="Arrow", GlyphTransform="Transform2" )',
			'\tG.GlyphTransform = "Transform2"',
			'\tG.Scalars = [\'POINTS\', zone]',
			'\tG.ScaleMode = \'scalar\'',
			'\tG.GlyphType = "Sphere"',
			'\tG.RandomMode = 0',
			'\tG.MaskPoints = 0',
			'\t',
			'\tG.GlyphType.Radius = %10.5f'%(2*ndRadius),
			'\t',
			'\tRenameSource(zone, G)',
			'\t',
			'\trv = GetRenderView()',
			'\tmr = GetDisplayProperties(source)',
			'\tdr = Show()',
			'\tdr.ColorArrayName = (\'POINT_DATA\', \'n\')',
			'\tdr.ScaleFactor = 1.1',
			'\tdr.SelectionPointFieldDataArrayName = zone',
			'\tdr.EdgeColor = [0.0, 0.0, 0.5000076295109483]',
			'\tdr.Opacity = 0.5',
			'\t',
			'\tlt = GetLookupTableForArray(zone, 1, RGBPoints=[0.0, 0.23, 0.299, 0.754, 0.5, 0.865, 0.865, 0.865, 1.0]+col, VectorMode=\'Magnitude\', NanColor=[0.25, 0.0, 0.0], ColorSpace=\'Diverging\', ScalarRangeInitialized=1.0 )',
			'\t',
			'\tpf = CreatePiecewiseFunction( Points=[0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0] )',
			'\t',
			'\tdr.ColorArrayName = (\'POINT_DATA\', zone)',
			'\tdr.LookupTable = lt',
			'\tdr.Visibility = 0',
			'\t',
			'\tlt.ScalarOpacityFunction = pf',
			]
		
		
		################################### load in contour output ###################################	
		if len(contour_files)>0:
			lns += ['contour_output = LegacyVTKReader( FileNames=[']
			for file in contour_files:
				file = file.replace('\\','/')
				lns += ['\''+file+'\',']
			lns += ['] )']
			lns += ['RenameSource("contour_output", contour_output)']
		
		################################### set up initial visualisation ###################################	
			dflt_cont = '\''+self.default_contour_variable+'\''		
			cont_lim = self.default_contour_lims
		
			viewTime = len(self.contour.times)-1
		
			lns += [
				'lt = GetLookupTableForArray('+dflt_cont+', 1, RGBPoints=[%10.5f, 0.23, 0.299, 0.754, %10.5f, 0.706, 0.016, 0.15], VectorMode=\'Magnitude\', NanColor=[0.25, 0.0, 0.0], ColorSpace=\'Diverging\', ScalarRangeInitialized=1.0 )'%tuple(cont_lim),
				'',
				'pf = CreatePiecewiseFunction( Points=[%10.5f, 0.0, 0.5, 0.0, %10.5f, 1.0, 0.5, 0.0] )'%tuple(cont_lim),
				'',
				'dr = Show() #dr = DataRepresentation1',
				'dr.Representation = \'Surface With Edges\'',
				'dr.EdgeColor = [0.15, 0.15, 0.15]',
				'dr.ScalarOpacityFunction = pf',
				'dr.ColorArrayName = (\'POINT_DATA\', '+dflt_cont+')',
				'dr.ScalarOpacityUnitDistance = 1.7320508075688779',
				'dr.LookupTable = lt',
				'',
				'rv.ViewTime = %4i'%viewTime,
				'',
				'ScalarBarWidgetRepresentation1 = CreateScalarBar( Title='+dflt_cont+', LabelFontSize=12, Enabled=1, LookupTable=lt, TitleFontSize=12 )',
				'GetRenderView().Representations.append(ScalarBarWidgetRepresentation1)',
				'',
				]
			
		if len(contour_files)>0:
			lns+= [
				'model = FindSource("model")',
				'model_rep = GetDisplayProperties(model)',
				'contour_output = FindSource("contour_output")',
				'cont_rep = GetDisplayProperties(contour_output)',
				]	
			if self.initial_show == 'material':
				lns+=[
					'model_rep.Visibility = 1',
					'cont_rep.Visibility = 0	',
					]					
			elif self.initial_show == 'contour':
				lns += [
					'model_rep.Visibility = 0',
					'cont_rep.Visibility = 1',
					]			
		
		if self.csv is not None:
			################################### load in history output ###################################	
			lns += ['xyview = CreateXYPlotView()']
			lns += ['xyview.BottomAxisRange = [0.0, 5.0]']
			lns += ['xyview.TopAxisRange = [0.0, 6.66]']
			lns += ['xyview.ViewTime = 0.0']
			lns += ['xyview.LeftAxisRange = [0.0, 10.0]']
			lns += ['xyview.RightAxisRange = [0.0, 6.66]']
			lns += ['']
			if os.path.isfile(self.csv.filename):
				lns += ['hout = CSVReader( FileName=[r\''+self.csv.filename+'\'] )']
				lns += ['RenameSource("history_output",hout)']
				
				fp = open(self.csv.filename)
				ln = fp.readline().rstrip()
				fp.close()
				headers = [lni for lni in ln.split(',')]
				
				# put variables in order to account for diff or time derivatives
				vars = []
				for variable in self.csv.history.variables:
					vars.append(variable)
					if self.csv.diff: vars.append('diff_'+variable)
					#if self.time_derivatives: vars.append('d'+variable+'_dt')
				for i,variable in enumerate(vars):
					plot_title = variable+'_history'
					lns += []
					lns += [plot_title+' = PlotData()']
					lns += ['RenameSource("'+plot_title+'",'+plot_title+')']
					
					lns += ['mr = Show()']
					lns += ['mr = GetDisplayProperties('+plot_title+')']
					lns += ['mr.XArrayName = \'time\'']
					lns += ['mr.UseIndexForXAxis = 0']
					lns += ['mr.SeriesColor = [\'time\', \'0\', \'0\', \'0\']']
					lns += ['mr.AttributeType = \'Row Data\'']
					switch_off = [header for header in headers if not header.strip().startswith(variable+':')]
					ln = 'mr.SeriesVisibility = [\'vtkOriginalIndices\', \'0\', \'time\', \'0\''
					for header in switch_off:
						ln+=', \''+header+'\',\'0\''
						
					#lns += ['mr.SeriesVisibility = [\'vtkOriginalIndices\', \'0\', \'time\', \'0\']']
					lns += [ln+']']
					if i != (len(vars)-1):
						lns += ['mr.Visibility = 0']
			
			lns += ['']
			lns += ['AnimationScene1.ViewModules = [ RenderView1, SpreadSheetView1, XYChartView1 ]']
			
			lns += ['Render()']
		
		
		######################################## load in well ########################################	
		if self.wells is not None:
			lns += ['model_rep.Representation = \'Outline\'']
			for well in self.wells:	
			
				lns += ['%s=LegacyVTKReader(FileNames=[r\'%s\'])'%(well,os.getcwd()+os.sep+well+'_wells.vtk')]
				lns += ['RenameSource("%s", %s)'%(well,well)]
				lns += ['SetActiveSource(%s)'%well]
				lns += ['dr = Show()']
				lns += ['dr.ScaleFactor = 1366.97490234375']
				lns += ['dr.SelectionCellFieldDataArrayName = \'Name\'']
				lns += ['mr = GetDisplayProperties(%s)'%well]
				lns += ['mr.LineWidth=4.0']

		f.writelines('\n'.join(lns))
		f.close()
	def _get_filename(self): return self.path.absolute_to_file+os.sep+self.path.filename
	filename = property(_get_filename) #: (**)
class fcsv(object):
	def __init__(self,parent,filename,history,diff,time_derivatives):
		self.parent = parent
		self.path = fpath(parent = self)
		self.path.filename = filename
		self.data = None
		self.history = history
		self.diff = diff
		self.time_derivatives = time_derivatives
		if diff:
			self.assemble_diff()
		if time_derivatives:
			self.assemble_time_derivatives()
	def assemble_diff(self):
		for variable in self.history.variables:
			for node in self.history.nodes:
				self.history.new_variable('diff_'+variable,node,self.history[variable][node]-self.history[variable][node][0])
	def assemble_time_derivatives(self):
		return
		#for variable in self.history.variables:
		#	for node in self.history.nodes:
		#		data = self.history[variable][node]
		#		time = self.history.times
		#		self.history.new_variable('d'+variable+'_dt',node,dt)
		#		
		#			ind = np.where(time==self.contour.times)[0][0]
		#			if ind == 0: 
		#				# forward difference
		#				dt = self.contour.times[1]-time
		#				f0 = self.contour[time][var]
		#				f1 = self.contour[self.contour.times[1]][var]
		#				dat = (f1-f0)/dt
		#			elif ind == (len(self.contour.times)-1): 
		#				# backward difference
		#				dt = time-self.contour.times[-2]
		#				f0 = self.contour[self.contour.times[-2]][var]
		#				f1 = self.contour[time][var]
		#				dat = (f1-f0)/dt
		#			else:
		#				# central difference
		#				dt1 = time - self.contour.times[ind-1]
		#				dt2 = self.contour.times[ind+1] - time
		#				f0 = self.contour[self.contour.times[ind-1]][var]
		#				f1 = self.contour[time][var]
		#				f2 = self.contour[self.contour.times[ind+1]][var]
		#				dat = -dt2/(dt1*(dt1+dt2))*f0 + (dt2-dt1)/(dt1*dt2)*f1 + dt1/(dt2*(dt1+dt2))*f2
		#			self.data.contour[time].append(pv.Scalars(dat,name='d_'+var+'_dt',lookup_table='default'))
					
	def write(self):	
		"""Call to write out csv files."""
		if self.parent.work_dir: wd = self.parent.work_dir
		else: wd = self.parent._path.absolute_to_file
		if wd is None: wd = ''
		else: wd += os.sep
		
		# write one large .csv file for all variables, nodes
		from string import join
		fp = open(wd+self.path.filename,'w')
		self.filename = wd+self.path.filename
		# write headers
		ln = '%16s,'%('time')
		vars = []
		for variable in self.history.variables:
			vars.append(variable)
			if self.diff: vars.append('diff_'+variable)
			#if self.time_derivatives: vars.append('d'+variable+'_dt')
		for variable in vars:
			for node in self.history.nodes:		
				var = variable
				#if len(var)>6: var = var[:6]
				ln += '%16s,'%(var+': nd '+str(node))
		fp.write(ln[:-1]+'\n')
		
		# write row for each time
		for i,time in enumerate(self.history.times): 		# each row is one time output
			ln = '%16.8e,'%time
			for variable in vars:
				for node in self.history.nodes:			# each column is one node
					ln += '%16.8e,'%self.history[variable][node][i]
			ln = ln[:-1]+'\n'
			fp.write(ln)
		fp.close()		
	
def fdiff( in1, in2, format='diff', times=[], variables=[], components=[], nodes=[]):
	'''Take the difference of two fpost objects
	
	:param in1: First fpost object
	:type filename: fpost object (fcontour)
	:param in2: First fpost object
	:type filename: fpost object (fcontour)
	:param format: Format of diff: diff->in1-in2 relative->(in1-in2)/abs(in2) percent->100*abs((in1-in2)/in2)
	:type format: str
	:param times: Times to diff
	:type times: lst(fl64)
	:param variables: Variables to diff
	:type variables: lst(str)
	:param components: Components to diff (foutput objects)
	:type components: lst(str)
	:returns: fpost object of same type as in1 and in2
	'''
	
	# Copy in1 and in2 in case they get modified below
	in1 = deepcopy(in1)
	in2 = deepcopy(in2)
	if type(in1) is not type(in2):
		print("ERROR: fpost objects are not of the same type: "+str(type(in1))+" and "+str(type(in2)))
		return
	if isinstance(in1, fcontour) or isinstance(in1, fhistory) or 'foutput' in str(in1.__class__):
		# Find common timesclear
		t = np.intersect1d(in1.times,in2.times)
		if len(t) == 0:
			print("ERROR: fpost object times do not have any matching values")
			return
		if len(times) > 0:
			times = np.intersect1d(times,t)
			if len(times) == 0:
				print("ERROR: provided times are not coincident with fpost object times")
				return
		else:
			times = t
			
	if isinstance(in1, fcontour):
		# Find common variables
		v = np.intersect1d(in1.variables,in2.variables)
		if len(v) == 0:
			print("ERROR: fcontour object variables do not have any matching values")
			return
		if len(variables) > 0:
			variables = np.intersect1d(variables,v)
			if len(variables) == 0:
				print("ERROR: provided variables are not coincident with fcontour object variables")
				return
		else:
			variables = v
		out = deepcopy(in1)
		out._times = times
		out._variables = variables
		out._data = {}
		for t in times:
			if format is 'diff':
				out._data[t] = dict([(v,in1[t][v] - in2[t][v]) for v in variables])
			elif format is 'relative':
				out._data[t] = dict([(v,(in1[t][v] - in2[t][v])/np.abs(in2[t][v])) for v in variables])
			elif format is 'percent':
				out._data[t] = dict([(v,100*np.abs((in1[t][v] - in2[t][v])/in2[t][v])) for v in variables])
		return out
	
	#Takes the difference of two fhistory objects.	
	elif isinstance(in1, fhistory):
		# Find common variables
		v = np.intersect1d(in1.variables, in2.variables)
		if len(v) == 0:
			print("ERROR: fhistory object variables do not have any matching values")
			return
		if len(variables) > 0:
			variables = np.intersect1d(variables,v)
			if len(variables) == 0:
				print("ERROR: provided variables are not coincident with fhistory object variables")
				return
		else:
			variables = v
			
		#Find common nodes.
		n = np.intersect1d(in1.nodes, in2.nodes)
		if len(n) == 0:
			print("ERROR: fhistory object nodes do not have any matching values")
			return
		if len(nodes) > 0:
			nodes = np.intersect1d(nodes,n)
			if len(nodes) == 0:
				print("ERROR: provided nodes are not coincident with fhistory object nodes")
				return
		else:
			nodes = n
		
		#Set up the out object.
		out = deepcopy(in1)
		out._times = times
		out._variables = variables
		out._nodes = nodes
		out._data = {}
		
		#Find the difference at each time index for a variable and node.
		for v in variables:
			for n in nodes:
				i = 0
				diff = []
				while i < len(times):
					if format is 'diff':
						#Quick fix to handle ptrk files.
						if isinstance(in1, fptrk):
							diff.append(in1[v][n]-in2[v][n]) 
						else:
							diff.append(in1[v][n][i]-in2[v][n][i])	
					elif format is 'relative':
						diff.append((in1[t][v] - in2[t][v])/np.abs(in2[t][v])) 
					elif format is 'percent':
						diff.append(100*np.abs((in1[t][v] - in2[t][v])/in2[t][v])) 	
					i = i + 1
				if isinstance(in1, fptrk):
					out._data[v] = np.array(diff)
				else:
					out._data[v] = dict([(n, diff)])
				
		#Return the difference.		
		return out
	  		
	elif 'foutput' in str(in1.__class__):
		# Find common components
		c = np.intersect1d(in1.components,in2.components)
		if len(c) == 0:
			print("ERROR: foutput object components do not have any matching values")
			return
		if len(components) > 0:
			components = np.intersect1d(components,c)
			if len(components) == 0:
				print("ERROR: provided components are not coincident with foutput object components")
				return
		else:
			components = c		
		out = deepcopy(in1)
		out._times = times
		out._node = {}
		for tp in ['water','gas','tracer1','tracer2']:		
			out._node[tp] = None
		for cp in components:
			if format is 'diff':
				if len(variables):
					out._node[cp] = dict([(n,dict([(v,np.array(in1._node[cp][n][v]) - np.array(in2._node[cp][n][v])) for v in list(in1._node[cp][n].keys()) if v in variables])) for n in in1.nodes])
				else:
					out._node[cp] = dict([(n,dict([(v,np.array(in1._node[cp][n][v]) - np.array(in2._node[cp][n][v])) for v in list(in1._node[cp][n].keys())])) for n in in1.nodes])
			elif format is 'relative':
				if len(variables):
					out._node[cp] = dict([(n,dict([(v,(np.array(in1._node[cp][n][v]) - np.array(in2._node[cp][n][v]))/np.abs(in2._node[cp][n][v])) for v in list(in1._node[cp][n].keys()) if v in variables])) for n in in1.nodes])
				else:
					out._node[cp] = dict([(n,dict([(v,(np.array(in1._node[cp][n][v]) - np.array(in2._node[cp][n][v]))/np.abs(in2._node[cp][n][v])) for v in list(in1._node[cp][n].keys())])) for n in in1.nodes])
			elif format is 'percent':
				if len(variables):
					out._node[cp] = dict([(n,dict([(v,100*np.abs((np.array(in1._node[cp][n][v]) - np.array(in2._node[cp][n][v]))/in2._node[cp][n][v])) for v in list(in1._node[cp][n].keys()) if v in variables])) for n in in1.nodes])
				else:
					out._node[cp] = dict([(n,dict([(v,100*np.abs((np.array(in1._node[cp][n][v]) - np.array(in2._node[cp][n][v]))/in2._node[cp][n][v])) for v in list(in1._node[cp][n].keys())])) for n in in1.nodes])
						
		return out
		
def sort_tec_files(files):
	# sort first by number, then by type
	for file in files:
		if not file.endswith('.dat'): return files
	paths = [os.sep.join(file.split(os.sep)[:-1]) for file in files]
	files = [file.split(os.sep)[-1] for file in files]
	
	times = []
	for file in files: 
		for type in ['_days_sca_node','_days_vec_node','_days_hf_node','_days_con_node','_sca_node','_vec_node','_hf_node','_con_node']:
			if type in file: 				
				times.append(float('.'.join(file.split(type)[0].split('.')[1:])))
				break
	times = sorted(enumerate(times), key=lambda x: x[1])
	
	paths = [paths[ind] for ind,time in times]
	files = [files[ind] for ind,time in times]

	return [path+os.sep+file if path else file for path,file in zip(paths,files)]
	
	
