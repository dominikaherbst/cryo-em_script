#!/usr/bin/env python
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                 written by Dominik A. Herbst                       #
#                     dherbst@berkeley.edu                           #
#             Usage without guarantees or warranties!                #
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#


import sys, os, copy
import numpy as np
from pprint import pprint
import relion_metadata_labels as meta
import re
import warnings
from io import StringIO
import numpy.lib.recfunctions as rf





##################################################################################
############################## STARFILE CLASS START ##############################

class starfile():
	
	def __init__(self, star_inp, verbosity=False, objname=None):
		
		self.star_inp = star_inp
		self.verbosity = verbosity
		self.objname=objname
		self.default_string_dtype = 'U1000' 
		self.data_len=0
		self.optics_len=0
		self.len_screen_header_for_data_blocks = None 
		# dictionary according to Relion-3.1
		self.assign_dtype = meta.relion3_1(self.default_string_dtype)
		
		self.data_block_names = [] # list of all data block names in the star file
		self.read_star_file(star_inp) # fills self.data_opt and self.data_ptcls
		
	
	def __str__(self):
		if self.objname is None: return "Object has no name"
		else: return self.objname
	
	
	def verbose(self, message, pp=False):
		if self.verbosity: 
			if pp: pprint(message)
			else: print(message)
	
	def read_star_file(self, fname):
		# return: 														colum_positions, colum_positions_inv, dtype_assignment, data_array
		# colum_positions:	 	dict[col_names str]=col_number int		dictionary, which contains the column names as key and the column numer (starting with 1) as element
		# colum_positions_inv:	dict[col_number int]=col_names str		make second dict - but keys and elements inverted => colnum : "colname"
		# dtype_assignment:		arr[int]=(col_names str, data type)		data type of each column
		# data_array													structured array containing all data without header
		#last_header_row												last_header_row
		
		
		##### Determine how many loop blocks exist
		
		# Determine how many data blocks are in the file
		starfile_all_lines = self.readfile(fname)
		data_line_num = self.find_line_in_file_starts_with(starfile_all_lines[:self.len_screen_header_for_data_blocks], "data")
		loop_line_num = self.find_line_in_file_starts_with(starfile_all_lines[:self.len_screen_header_for_data_blocks], "loop_")
		column_header_lines = self.find_line_in_file_starts_with(starfile_all_lines[:self.len_screen_header_for_data_blocks], "_")
		
		
		if len(data_line_num) != len(loop_line_num): sys.exit("Cannot read starfile with data blocks containing tables and loops!")
		
		########## READ DATA BLOCKS
		
		num_data_block = len(loop_line_num)
		
		column_header_lines_idx_count= 0
		for block_idx,(b_line_num,block_name) in enumerate(data_line_num):
			self.verbose("-------------------------------------------------" )
			self.verbose("Reading data block %d (%s)" % (block_idx+1, block_name) )
			first_col_label_line_idx=loop_line_num[block_idx][0]+1
			
			############# 
			# Create dictionary with column name as key and column number as element e.g. --> colum_positions["_rlnBeamTiltX"] : 11
			# dictionary, which contains the column names as key and the column numer (starting with 1) as element
			last_line = None # this is last line in the star with a column meta label for the next data block
			colum_positions = {}
			for col_idx, (starfile_line_num, col) in enumerate(column_header_lines[column_header_lines_idx_count:]):
				#print "column_header_lines_idx_count=", column_header_lines_idx_count , "\tcol_idx=", col_idx
				if col_idx == 0 and first_col_label_line_idx != starfile_line_num: 
					#print col_idx , first_col_label_line_idx, starfile_line_num
					sys.exit("Is there an empty line between loop_ and the first column label of data block %s?" % block_name)
				if isinstance(last_line, int) and last_line+1 != starfile_line_num: #detect a jump in the starfile num ... if it is from the same block you would expect that last_line +1 == starfile_line_num ... in unequal, the block is over!
					break
				last_line = starfile_line_num
				beginning = col.split(" ")[0]
				colum_positions[beginning] = col_idx+1
				column_header_lines_idx_count += 1 
			
			###########
			# Create the inverse dictionary e.g. --> colum_positions_inv[11] : "_rlnBeamTiltX"
			colum_positions_inv = { colum_positions[i] : i  for i in colum_positions} # make second dict - but keys and elements inverted => colnum : "colname"
			
			##########
			# Create dictionary that connects column names with data type
			dtype_assignment = []
			for i in range(1, len(colum_positions_inv)+1):
				try: dtype_assignment.append( (colum_positions_inv[i], self.assign_dtype[colum_positions_inv[i]]) )
				except KeyError: dtype_assignment.append( (colum_positions_inv[i], self.default_string_dtype ) )
			
			self.verbose( "Column headers and data types assigned to columns:")
			self.verbose( "column     data type     column name")
			for idx, (colname,dtype) in enumerate(dtype_assignment): self.verbose(  "    %02d     % 9s     %s" % (idx+1, dtype, colname) )
			
			
			
			
			#########
			# Read data arrays
			# determine skip footer --> get line where next data block starts
			# if there is no next data block 
			#try: skip_footer=self.lines_in_data_star-data_line_num[block_idx+1][0] -1 #its over where the next data block starts (=line number in star file where next data block starts)
			#except IndexError: skip_footer=0
			
			
			
			try: end_datablock_line=data_line_num[block_idx+1][0]
			except IndexError: end_datablock_line=None
			
			self.verbose( "Reading array...")
			data_array = np.genfromtxt(StringIO("".join(starfile_all_lines[last_line+1:end_datablock_line])), 
				dtype=dtype_assignment, 
				comments='#'
			) # structured array using the dtype assignment + column names from above
			#print "Data read in:\n", data_array # structured array = can be called by column names e.g. data_array["_rlnDefocusU"]
			
			
			
			self.verbose("%i data elements imported." % ( data_array.size ))
			
			if data_array.size == 1: data_array = np.atleast_1d(data_array)
			
			
			#if num_data_block == 1 or (num_data_block > 1 and "particles" in block_name):
			#	try: data_len = len(data_array)
			#	except TypeError: sys.exit("ERROR: %s must contain at least two particles!" % fname)
			## TODO: UNRESOLVED PROBLEM: only one group! --> 1D array causes problems!!!
			##elif num_data_block == 1 or (num_data_block > 1 and "optics" in block_name):
			##	try: optics_len = len(data_array)
			##	except TypeError: sys.exit("ERROR: %s must contain at least one optics group!" % fname)
			
			
			##### returns:
			# data_array
			# colum_positions
			# colum_positions_inv
			# dtype_assignment
			
			
			
			
			########################## GENERATE NEW CLASS HERE DON'T USE DICTIONARY !!!! INHERIT CLASS ETC ...
			###### GENERATE A CLASS with the name of the data block, e.g. data_particles and assign the subclass data_block
			# it will be accessible via:
			# self.data_particles.data_array
			# self.data_particles.dict_colnum_colname
			# self.data_particles.dict_colname_colnum
			# self.data_particles.arr_col_dtype_assignment
			# self.data_optics.arr_col_dtype_assignment
			# ....
			# self.data_random_data_table.data_array
			data_block_name = block_name.replace(" ", "") # remove spaces (e.g. "data_optics" or "data_" or "data_particles")
			if data_block_name in self.data_block_names: 
				print("WARNING: %s contains data blocks with identical name! %s was renamed to %s" % ( fname, data_block_name, data_block_name + "_block" + str(block_idx+1) ))
				data_block_name = data_block_name + "_block" + str(block_idx+1)
			else: print("Data block object created: %s" % data_block_name)
			self.data_block_names.append(data_block_name)
			setattr(self, data_block_name, data_block(data_array, colum_positions_inv, colum_positions, dtype_assignment, objname=data_block_name))
			
			self.verbose("-------------------------------------------------")
			#sys.exit()
			#pprint ( data_array)
		
		
		#if num_data_block == 1: return data_blocks[data_blocks.keys()[0]]
		#else: 
		#return data_blocks
		#OLD: return colum_positions, colum_positions_inv, dtype_assignment, data_array, last_header_row
		
	def copy_data_block(self, old, new):
		from copy import deepcopy
		obj = getattr(self, str(old))
		setattr(self, str(new), deepcopy(obj))
		self.data_block_names.append(new)
	
	
	
	def find_line_in_file_starts_with(self, file_handle, search_str):
		### file_handle = self.readfile(filename, length=200)
		# searches for a string in a file and returns the line numbers as list starting at 0
		#match_line = []
		return [ (line_num, line.replace("\n", "")) for line_num,line in enumerate(file_handle) if re.match(r'^%s' % search_str, line) ] 
	
	def readfile (self, filename, length=None):
		try:
			handle = open(filename,"r")
		except:
			print("ERROR: Do you have permission to read %s ?" % filename)
			sys.exit(0)
		
		lines = handle.readlines()
		self.lines_in_data_star = len(lines)
		handle.close()
		return lines[0:length]
	
	def strip_end(self, string, suffix):
		# remove suffix from string, if suffix exists
		if not string.endswith(suffix):
			return string
		return string[:len(string)-len(suffix)]
	
	def getattr(self, objname):
		return getattr(self, objname)
	
	
	def savestar(self, fileout, data_blocks_list=None, reset_col=False):
		# data_blocks_list		(list) names of the data blocks to write. Default: (None type) = all
		# reset_col				(bool) if set then the column write list will be reset and all available columns will be written. 
		#						This is usefull if columns have been modified after a starfile has been saved.
		#### EXPLAIN:
		# In order to write a selection of colums for each data block, use the methods 
		# self.data_blockname.write_exclude_column(column1, column2, column3, ....) in order to exclude these columns
		# or
		# self.data_blockname.write_include_column(column1, column2, column3, ....) in order to include these columns. In this case only these columns will be written.
		# These methods have to be executed before savestar is executed!!
		#
		#
		# In order to write just a/some specific data block(s), provide a list with the data block names when calling this method.
		
		self.verbose("-------------------------------------------------")
		if data_blocks_list is None: data_blocks_list = self.data_block_names
		else: 
			for i in data_blocks_list:
				if i not in self.data_block_names: sys.exit("ERROR: Data block %s does not exist!") 
		f = open(fileout, "w")
		for blockname in data_blocks_list:
			
			block = getattr(self, blockname)
			self.verbose( "Preparing to write %s " % block )
			
			####### generate meta data header for block:
			columns2write = list(block.make_write_column_list(reset=reset_col)) # list
			header="%s\n\nloop_\n" % blockname
			cols = [ "%s #%i" % (colname, idx+1) for idx,colname in enumerate(columns2write) ]
			header += "\n".join(cols)
			
			####### generate dtype string for data:
			fmt_data_arr = []
			for name in columns2write: 
				try: dtype = block.dict_colname_dtype[name]
				except KeyError:  dtype = self.default_string_dtype
				fmt_data_arr.append(self.dtype_one_letter_to_formating_str(dtype))
				#else: fmt_data_arr.append('%s')
			fmt_data_str = "\t".join(fmt_data_arr)
			
			####### column selection to write:
			self.verbose("Columns to write:")
			self.verbose(columns2write, pp=True)
			s = StringIO()
			#print(block.data_array[columns2write])
			write_array = rf.repack_fields(block.data_array[columns2write]) #### In some numpy version there is a problem with views: '1.16.2' ... Indexing works differently ... so repack
			#print block.data_array[columns2write]
			#print columns2write
			#print "list(block.data_array.dtype.names)"
			#print list(block.data_array.dtype.names)
			
			#print(fmt_data_str)
			np.savetxt(s, write_array[columns2write], fmt=fmt_data_str, header=header, comments='')
			f.write(s.getvalue()+"\n\n")
			self.verbose("%d elements saved in data block %s" % (len(write_array),blockname))
		f.close()
		self.verbose("File saved: %s" % fileout)
	
	def dtype_one_letter_to_formating_str(self, dtype):
		d = { 
			'i' : '%d',
			'f' : '%06f',
			'b' : '%r',
			self.default_string_dtype: '%s'
		}
		try: return d[dtype]
		except KeyError: 
			print("WARNING: dtype (%s) not identified! Using default: %s " % (dtype, "%s"))
			return "%s"


class data_block():
	
	
	def __init__(self, data_array, dict_colnum_colname, dict_colname_colnum, arr_col_dtype_assignment, objname=None):
		self.data_array					= data_array
		self.dict_colnum_colname		= dict_colnum_colname
		self.dict_colname_colnum		= dict_colname_colnum
		self.arr_col_dtype_assignment	= arr_col_dtype_assignment
		self.dict_colname_dtype			= { colname : dtype for colname, dtype in arr_col_dtype_assignment }
		self.objname					= objname
		self.default_string_dtype 		= "S1000" #### NOTE: THIS EXISTS TWICE!!! ALSO IN STARFILE CLASS ... inherit didn't work, calling from other class did not work ...
		self.write_column_list			= []
	
	
	def __str__(self):
		if self.objname is None: return "Object has no name"
		else: return self.objname
	
	
	def write_exclude_column(self, *args):
		# if the write-out list is empty --> pre-fill it with all columns (sorted by the original column number to preserve the order)
		if len(self.write_column_list) == 0 : self.write_column_list = [ colname for colname, colnum in sorted(list(self.dict_colname_colnum.items()), key=lambda item: item[1]) ]
		# remove all given columns from the list
		[ self.write_column_list.remove(i) for i in args if self.check_colname_exists(i) ]
	
	
	def write_include_column(self, *args):
		# if the write-out list is empty --> fill it with all given columns 
		if len(self.write_column_list) == 0 : self.write_column_list = [ i for i in args if self.check_colname_exists(i) ]
		else: self.write_column_list += [ i for i in args if self.check_colname_exists(i) and i not in self.write_column_list ]
	
	
	def make_write_column_list(self, reset=False):
		if reset: self.write_column_list = []
		# if write_column_list has been edited before, just return it
		# otherwise: if the write out list is empty --> pre-fill it with all columns (sorted by the original column number to preserve the order)
		if len(self.write_column_list) > 0 : return self.write_column_list
		else: 
			self.write_column_list = [ colname for colname, colnum in sorted(list(self.dict_colname_colnum.items()), key=lambda item: item[1]) ]
			return self.write_column_list
	def purge_write_column_list(self):
		self.write_column_list = []
	
	
	def check_colname_exists(self, colname):
		if colname in self.data_array.dtype.names: return True
		else: False
	
	
	def add_column(self, value, column_name=None):
		# value:		constant (str, float, int, bool, np.array)
		#				If value is an array
		#				- it must have the same shape as the data_array (same amount of rows)
		#				- it can contain several colums, but in that case it must be a structured array with assigned column names
		#				- a regular ndarray can be provided together with a column_name if it is one dimensional
		# column_name:	Name of the new column (str)
		#
		# Column names must have a leading underscore and should be without spaces (will be corrected automatically). 
		#
		if column_name is not None: column_name = self.leading_underscore(column_name)
		if type(value) == np.ndarray:  # value is an array with the same shape as the data (same amount of rows)
			#If no column name is provided, check whether the array has already a name!!
			if value.shape != self.data_array.shape:  raise Exception("ERROR: Array must have the same length!")
			
			if column_name is None and value.dtype.names is None: # check if it is a structured array and has already a column name
				 raise Exception("ERROR: Either provide a column name or provide a structured array that already has a name!")
			elif column_name is not None and value.dtype.names is None: 
				if value.ndim > 1: raise Exception("ERROR: You cannot provide a multidimensional array that is not structured with only one column name! If you want to add several columns you have to add a structured array that has already column names!")
				value.dtype=[( self.leading_underscore(column_name) , value.dtype[0].str)]
			else: # A structured array was provided (has already a column name(s))
				if column_name is not None : # This can be only a structured array with 1 column
					if len(value.dtype.names) > 1: raise Exception("Sorry, cannot rename several columns with one column name!")
					verbose("The preexisting column name (%s) will be renamed to %s." % (value.dtype.names[0], self.leading_underscore(column_name)))
					value.dtype.names = (self.leading_underscore(column_name),) # rename the column if an extra column name was provided
				else: ### This can be a structured array with several columns
					value.dtype.names = tuple([self.leading_underscore(n) for n in value.dtype.names])
			
			if len(value.dtype.names) > 1 : ### adding multidimensional startured array!
				verbose("You provided a structured array with more than one column: %s " % ",".join(list(value.dtype.names)))
				
				for idx,n in enumerate(value.dtype.names): 
					if self.check_colname_exists(n):  raise Exception("ERROR: The new column name %s already exists!" % n)
					elif n == '':  raise Exception("ERROR: The new column name %s is empty!" % n)
					### add column to dict_colname_colnum and dict_colnum_colname
					self.dict_colname_colnum[str(n)] = max(self.dict_colname_colnum.values())+1
					self.dict_colnum_colname[self.dict_colname_colnum[str(n)]] = str(n)
					self.dict_colname_dtype[str(n)] = self.arr_dtype_to_string_letter(value.dtype[idx].str)
				dtype_descr = value.dtype.descr # e.g. [('LALALA', '<i8')]
			elif len(value.dtype.names) == 1: # A 1 column structured array was provided (has already a column name)
				dtype_descr = value.dtype.descr  # e.g. [('LALALA', '<i8'), ('LULULU', '<f4')]]
				### add column to dict_colname_colnum and dict_colnum_colname
				self.dict_colname_colnum[column_name] = max(self.dict_colname_colnum.values())+1
				self.dict_colnum_colname[self.dict_colname_colnum[ column_name ]] = column_name
				self.dict_colname_dtype[ column_name ] = self.arr_dtype_to_string_letter(value.dtype[0].str)
			else: raise Exception("Something is weird with the array that you try to add as column!")
			new_col = value
		
		else: # value is a constant and a column with this constant will be added
			if column_name is None: raise Exception("You didn't provide a column name!")
			if type(value) is str: new_col = np.array([value]*len(self.data_array), dtype=[( column_name, self.default_string_dtype )])
			else: new_col = np.full(self.data_array.shape, value, dtype=[( column_name, type(value) )]) #### This will vail if it is a string (without raising an error!)
			dtype_descr = new_col.dtype.descr  
			self.dict_colname_colnum[str(column_name)] = max(self.dict_colname_colnum.values())+1
			self.dict_colnum_colname[self.dict_colname_colnum[str(column_name)]] = str(column_name)
			self.dict_colname_dtype[str(column_name)] = self.arr_dtype_to_string_letter(new_col.dtype[0].str)
		
		### to add the new column(s) we have to create a new (empty) array with all columns and rows:
		new_arr = np.zeros(
			self.data_array.shape, 
			dtype=self.data_array.dtype.descr+new_col.dtype.descr
		)
		# now fill it
		new_arr[list(self.data_array.dtype.names)][:] = self.data_array # old data
		new_arr[list(new_col.dtype.names)][:] = new_col # new column
		self.data_array = new_arr # overwrite old array with new array containing the new column(s)
		del(new_col) # clean up memory, usefull if new_col is large!
		del(new_arr) # clean up memory, usefull if new_arr is large!
		
	def del_columns(self, *columns):
		columns = [ self.leading_underscore(c) for c in columns]
		#delete from data_array:
		for c in columns: 
			if not self.check_colname_exists(c): raise ValueError("Column %s cannot be deleted, because it does not exist!" % c) 
		new_column_selection = [ c for c in self.data_array.dtype.names if c not in columns ]
		self.data_array = rf.repack_fields(self.data_array[new_column_selection])
		#delete from dictionaries:
		for c in columns: 
			if c in list(self.dict_colname_colnum.keys()): 
				del( self.dict_colnum_colname[self.dict_colname_colnum[c]] )
				del( self.dict_colname_colnum[c] )
			if c in list(self.dict_colname_dtype.keys()): del( self.dict_colname_dtype[c] )
	
	def rename_column(self, column_name_old, column_name_new):
		column_name_new=self.leading_underscore(column_name_new)
		column_name_old=self.leading_underscore(column_name_old)
		if self.check_colname_exists(column_name_new): raise Exception("ERROR: New colname %s already exists" % column_name_new)
		if column_name_old not in self.data_array.dtype.names: raise Exception("ERROR: Old colname %s does not exists" % column_name_old)
		self.data_array.dtype.names = tuple([ column_name_new if n == column_name_old else n for n in self.data_array.dtype.names])
		#update other dicts:
		self.dict_colname_colnum[column_name_new] = self.dict_colname_colnum.pop(column_name_old)
		self.dict_colnum_colname[self.dict_colname_colnum[column_name_new]] = column_name_new
		self.dict_colname_dtype[ column_name_new ] = self.dict_colname_dtype.pop(column_name_old)

		
	#def check_colname_exists(self, colname):
	#	#check whether the given colname exists in the data block, e.g. for writing a datablock selectively to disk / typo check
	#	if colname in self.dict_colname_colnum.keys(): return True
	#	else: return False
	
	def arr_dtype_to_string_letter(self, dtype):
		d = {
			'<f4' : 'f',
			'<f8' : 'f',
			'|S1000' : self.default_string_dtype,
			'<i8' : 'i',
			'<i4' : 'i',
			'|b1' : 'b'
		}
		try: return  d[dtype]
		except KeyError: 
			print("WARNING: dtype (%s) not identified! Using default: %s " % (dtype, self.default_string_dtype))
			return self.default_string_dtype
	
	#def replace_column(self, col):
	#	# not required!!
	#	# its a simple: datafile.data_optics.data_array["_rlnBeamTiltY"] = np.zeros(datafile.data_optics.data_array["_rlnBeamTiltY"].shape)
	#	pass
	
	
	def swap_column_positions(self, col1, col2):
		col1 = self.leading_underscore(col1)
		col2 = self.leading_underscore(col2)
		if not self.check_colname_exists(col1): raise Exception("%s does not exist!" % col1)
		if not self.check_colname_exists(col2): raise Exception("%s does not exist!" % col2)
		
		colnum_col1 = self.dict_colname_colnum[col1]
		colnum_col2 = self.dict_colname_colnum[col2]
		self.dict_colname_colnum[col1] = colnum_col2
		self.dict_colname_colnum[col2]= colnum_col1
		
		colname_col1 = self.dict_colnum_colname[colnum_col1]
		colname_col2 = self.dict_colnum_colname[colnum_col2]
		self.dict_colnum_colname[colnum_col1]=colname_col2
		self.dict_colnum_colname[colnum_col2]=colname_col1
		
		dcoln = list(self.data_array.dtype.names)
		idxc1 = dcoln.index(col1)
		idxc2 = dcoln.index(col2)
		dcoln[idxc1], dcoln[idxc2] = dcoln[idxc2], dcoln[idxc1]
		self.data_array = self.data_array[dcoln]
		
	
	def column_set_constant(self, column, value):
		column = self.leading_underscore(column)
		if not self.check_colname_exists(column): raise ValueError("Column %s does not exist!" % column)
		else:
			dtype = self.data_array.dtype.fields[column][0]
			col_old = self.data_array[[column]]
			
			if type(value) is str: new_col = np.array([value]*len(self.data_array), dtype=[( column, dtype )])
			else: new_col = np.full(self.data_array.shape, value, dtype=[( column, dtype )]) #### This will vail if it is a string (without raising an error!)
			self.data_array[[column]] = new_col
			return new_col
	
	def leading_underscore(self, test):
		test=str(test).replace(" ", "_")
		if test.startswith("_"): return test
		else: return "_"+test
	
	def random_select_sample(self, num=None, replace=False, overwrite=False):
		# reduces the data block to a specified amount (num) of random ptcls
		# num		(int)	amount of ptcls that should remain
		# replace	(bool)	if True a random sample with replacement will be returned
		# overwrite	(bool)	if True the data block will be overwritten with the random sample. If False the funtion solely returns the random array
		np.random.seed() # important if the function will be called from a parallel instance (--> same time --> same random seed --> same selection)
		if num is None: num = len(self.data_array)
		random_sample = np.random.choice(self.data_array, int(num), replace=replace)
		if overwrite: self.data_array = random_sample
		else: return random_sample
	





############################### STARFILE CLASS END ###############################
##################################################################################

def fields_view(arr, fields):
    dtype2 = np.dtype({name:arr.dtype.fields[name] for name in fields})
    return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)


def verbose(message, verbosity=True, pp=False):
	if verbosity: 
		if pp: pprint(message)
		else: print(message)

def add_leading (string, prefix):
	# remove suffix from string, if suffix exists
	if string.startswith(prefix): return string
	else: return prefix+string



def write2file (filename, content):
	try:
		handle = open(filename,"w")
	except:
		print("ERROR: Do you have the permission to write in this folder?")
		sys.exit(0)
	handle.writelines(content)
	handle.close()


def runcmd(cmd):
	import subprocess
	try: p = subprocess.Popen(cmd, shell=True)
	except: sys.exit("ERROR:\t could not execute %s" % cmd)
	p.wait()
	(process_out,process_err) = p.communicate()
	return process_out


def savetxt(filename, datablock):
	header="""  """
	np.savetxt(filename, np.transpose(datablock), fmt="%d\t%3.5f\t%3.5f", delimiter=' ', newline='\n', header=header, comments='')


















def euler2rot_ccp4(alpha, beta, gamma):
	# according to CCP4 convention
	ca = np.cos(alpha)
	cb = np.cos(beta)
	cg = np.cos(gamma)
	sa = np.sin(alpha)
	sb = np.sin(beta)
	sg = np.sin(gamma)
	r = np.array([[ca*cb*cg-sa*sg,	-ca*cb*sg-sa*cg,	ca*sb ],
					[sa*cb*cg+ca*sg,	-sa*cb*sg+ca*cg,	sa*sb],
					[-sb*cg,			sb*sg,				cb]])
	return r



def dynamo4ccp4_euler2rot(alpha,beta,gamma):
	### dynamo euler to matrix convention
	### radian as input
	### return as radian
	ca = np.cos(alpha)
	cb = np.cos(beta)
	cg = np.cos(gamma)
	sa = np.sin(alpha)
	sb = np.sin(beta)
	sg = np.sin(gamma)
	
	cc = cb*ca
	cs = cb*sa
	sc = sb*ca
	ss = sb*sa
	
	R = np.array([	[	cg*cc-sg*sa,	cg*cs+sg*ca,	-cg*sb ],
					[	-sg*cc-cg*sa,	-sg*cs+cg*ca,	sg*sb ],
					[	sc,	ss,	cb ] ]
				)
	return R

#
#def dynamo_rot2euler_old(R):
#	"""Decompose rotation matrix into Euler angles"""
#	### radian as input
#	### return as radian
#	tot=1e-4
#	
#	if abs(R[2,2]-1) < tot:
#		ALPHA=0
#		BETA=0
#		GAMMA=np.arctan2(R[1, 0], R[0, 0])
#	elif abs(R[2,2]+1) < tot:
#		ALPHA=0
#		BETA=180
#		GAMMA=np.arctan2(R[1, 0], R[0, 0])
#	else:
#		ALPHA = np.arctan2(R[2, 0], R[2, 1])
#		BETA = np.arccos(R[2,2])
#		GAMMA = np.arctan2(R[0,2],-R[1,2])
#	
#	return ALPHA, BETA, GAMMA
#

def dynamo_rot2euler(R):
	"""Decompose rotation matrix into Euler angles"""
	### radian as input
	### return as radian
	
	if R[2,2] < 1:
		BETA=np.arccos(R[2,2])
		ALPHA=np.arctan2(R[2, 1], R[2, 0])
		GAMMA=np.arctan2(R[1,2], -R[0,2])
	else:
		ALPHA = 0
		BETA = 0
		GAMMA = np.arctan2(R[0,1],R[1,1])
	
	return ALPHA, BETA, GAMMA





def apply_3D_coord_transform_to_ptcl_aln_params(AngleRot, AngleTilt, AnglePsi, OriginX, OriginY, apix, t_shift, eul, box_center=None):
	"""
	this version was upodated tu work with _rlnOriginXAngst and _rlnOriginYAngst, however, the variable still refer to the old 3.0 implementation with values in pixels!
	povide the column that refers to the shifts in angstroem
	------------------------------------
	Input parameters:
	AngleRot 	(nd-array float32, shape = (n,) )	= Original (unchanged) rlnAngleRot (Euler alpha) in deg, rotation around z
	AngleTilt 	(nd-array float32, shape = (n,) )	= Original (unchanged) rlnAngleTilt (Euler beta) in deg, rotation around y
	AnglePsi 	(nd-array float32, shape = (n,) )	= Original (unchanged) rlnAnglePsi (Euler gamma) in deg, rotation around z
	OriginX 	(nd-array float32, shape = (n,) )	= Original (unchanged) rlnOriginX (origin shift) in pixel
	OriginY 	(nd-array float32, shape = (n,) )	= Original (unchanged) rlnOriginY (origin shift) in pixel
	apix 		(float)								= Pixel size in Angstrom. Required for applying translations (in Angstrom)
	t_reconst 	(nd-array float32, shape = (3,) )	= Translation vector in Angstrom
														NOTE:	The origin for coordinate transformations is (0,0,0), while in EM it is the
																center of the box. For coordinate transformations the rotations are applied before
																translations, which rotates the molecule out of the box. The following translation is large
																in order to translate the molecule back in on its final position.
																In EM translations are applied before rotation (in order to center the projection for reconstruction).
																If the transformation is derived from coordinate transformations (e.g. from lsqkab or superpose / CCP4),
																the box_center parameter is required, which will be used to calculate the correct translation relative
																to the EM-box origin (center).
																Without the box_center parameter, translations are relative to the center.
	eul		 	(nd-array float32, shape = (3,) )	= Rotation Euler angles (alpha, beta, gamma) in deg, rotation: zyz around origin
														NOTE:	The origin for coordinate transformations in (0,0,0), while in EM it is the
																center of the box.
	box_center 	(nd-array float32, shape = (3,) )	= Coordinates in pixel for the center of the box. This parameter is required for calculating a corrected translation
														vector, if a transformation (rotation + translation) was derived from coordinate transformations (see above).
	"""
	
	if AngleRot.shape != AngleTilt.shape != AnglePsi.shape != OriginX.shape != OriginY.shape: sys.exit("Input alignment parameters must have the same shape!")
	
	try: n_ptcl = AngleRot.shape[0]
	except: n_ptcl = 1
	
	
	
	
	
	
	#### IMPORTANT - PARAMETERZATION
	#
	
	# Calculate Rotation_matrix from euler according to ccp4
	R_update=euler2rot_ccp4( *np.radians(eul) ) # unpack
	
	if box_center is not None:
		shift_box_adjusted = np.dot(R_update.T,t_shift) + box_center*apix - np.dot(R_update.T,box_center*apix)
		#print "New translation vector for rotations around the box center in voxels (box center = [%0.1f, %0.1f, %0.1f]): %5.3f, %5.3f, %5.3f" % (box_center[0],box_center[1],box_center[2], shift_box_adjusted[0], shift_box_adjusted[1], shift_box_adjusted[2])
		print("New translation vector for rotations around the box center in Angstrom (box center = [%0.1f, %0.1f, %0.1f]): %5.3f, %5.3f, %5.3f" % (box_center[0]*apix,box_center[1]*apix,box_center[2]*apix, shift_box_adjusted[0], shift_box_adjusted[1], shift_box_adjusted[2]))
	else: shift_box_adjusted = t_shift
	
	
	
	
	
	# Get rotation functions:
	#### IMPORTANT - PARAMETERZATION
	# dynamo_euler2rot required for correct parameterzation
	
	# get the rotation functions of all ptcls
	# sort the fucking axes:
	R_org = np.swapaxes ( dynamo4ccp4_euler2rot(np.radians( AngleRot ),np.radians( AngleTilt ),np.radians( AnglePsi )), 0, 1 ).T
	
	# outdated: # R_update = dynamo_euler2rot( *np.radians(eul) ) # unpack
	R_new = np.dot(R_org,R_update.T) # element wise multiplication = np.dot
	
	# convert angles back:
	if R_new.shape == (3,3): new_euler = np.array([ np.degrees( dynamo_rot2euler(R_new)) for i in np.arange(n_ptcl) ])
	else: new_euler = np.array([ np.degrees( dynamo_rot2euler(R_new[i])) for i in np.arange(n_ptcl) ])
	
	new_AngleRot  = new_euler[:,0]
	new_AngleTilt = new_euler[:,1]
	new_AnglePsi  = new_euler[:,2]
	
	
	# apply shift
	t_org = np.array([OriginX, OriginY, np.zeros((n_ptcl))]).T # shape = (n_ptcl, 3) 
	t_new = t_org + ( np.dot(R_org,shift_box_adjusted) )		# shift_box_adjusted.T = shift_box_adjusted (shape (3,)), not transposed, because t_org is transposed	
	
	new_OriginX = t_new[:,0]
	new_OriginY = t_new[:,1]
	
	
	
	# return
	return_as_matrix = np.vstack((
		new_AngleRot, 
		new_AngleTilt, 
		new_AnglePsi, 
		new_OriginX, 
		new_OriginY
	)).T
	
	return return_as_matrix



if __name__ == "__main__": print(0)
