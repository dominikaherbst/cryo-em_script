#!/usr/bin/env python
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                 written by Dominik A. Herbst                       #
#                     dherbst@berkeley.edu                           #
#             Usage without guarantees or warranties!                #
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

import sys, os, argparse
from pprint import pprint
import numpy as np
# add startools.py to your python path:
# export PYTHONPATH="$PYTHONPATH:/......"
import startools

sysmessage = \
""" 
-------------------------------------------------------------------------------
|                         coord_transform_to_star                             |
-------------------------------------------------------------------------------

v.3.0(210604)

This program can be used to apply a coordinate transformation (Euler angles, 
orthogonal translation in Angstrom) (e.g. from a superposition of two 
PDB files) to the alignment parameters of particles of a reconstruction. It can
be used to align a data set to a reference data set without particle alignment,
if the coordinate transformation of two fitted PDB models is known.
It can also be used to center particles of a reconstruction on a specific 
domain or in the box without new particle alignment.

The program reads and writes Relion 3.1 compatible (data) star files.

See -h --help for all options.

Written by Dominik A. Herbst (dherbst@berkeley.edu).

Usage without guarantees or warranties!
--------------------------------------------------------------------------------

"""
print(sysmessage)



def start_parser():
	# ---------------------- start parser ------------------------------------------
	parser = argparse.ArgumentParser(prog=os.path.basename(__file__), usage='%(prog)s [options]')
	parser.add_argument('-i', type=str, help='Input (data) star file with all ptcl.')
	parser.add_argument('-e', nargs=3, type=float, help='Euler angles (alpha, beta, gamma) according to Crowther with rotations around ZYZ (3D, e.g.: 30.0 10.0 0.0)')
	parser.add_argument('-t', nargs=3, type=float, help='Translation vector of the reconstruction in ANGSTROM (3D, e.g.: 0.0 10.0 20.0)')
	parser.add_argument('-o', type=str, default="transformed.star", help='Output filename. Default: [%(default)s]')
	parser.add_argument('-apix', type=float, default=1.0, help='Pixel size in Angstrom. Important to scale relative to coordinate transformations.')
	parser.add_argument('-box_center', nargs='+', type=float, help='Box center in PIXEL. Coordinate transformations derived from CCP4 programs refer to rotations around the origin (0,0,0), while the relion origin is in the center of the box. In order to use CCP4 coordinate transformations, provide the coordinates of the box center, e.g. 50.0 50.0 50.0 for a rectangular box with an endge length of 100 pixel.)')
	parser.add_argument('-v', action='store_const', const=True, default=False, help='Increase output verbosity')
	
	return parser.parse_args()
	# ------------------------------------------------------------------------------




def main():	
	variables = start_parser()
	print("Input parameters:")
	star_inp = variables.i
	print("star_inp = %s" % star_inp)
	
	euler = variables.e
	print("euler = %s " % euler)
	
	t = variables.t
	print("t = %s " % t)
	
	apix = variables.apix
	print("apix = %s " % apix)
	
	box_center = variables.box_center
	print("box_center = %s " % box_center)
	
	out_star = variables.o
	print("out_star = %s" % variables.o)
	
	if not os.path.isfile(star_inp): sys.exit("ERROR: %s does not exist!" % star_inp)
	
	
	
	
	# preprocessing of input parameters
	if star_inp is None: sys.exit("ERROR: Input star file must be provided!")
	if (t is not None) and (np.array(t).shape != (3,)): sys.exit("ERROR: Incorrect dimension! t must have three elements, e.g. 2.0 1.0 0.0 !")
	if (t is None) and (euler is None): sys.exit("Missing parameters! Nothing to do! ")
	if (euler is not None) and (np.array(euler).shape != (3,) ): sys.exit("ERROR: Incorrect dimension! Provide three Euler angles, e.g. 2.0 1.0 0.0")
	if euler is None: euler = np.array([0.0, 0.0, 0.0])
	else: euler = np.array(euler)
	if t is None: t = np.array([0.0, 0.0, 0.0])
	else: t = np.array(t)
	
	if ( box_center is not None ) and (np.array(box_center).shape != (3,) ):
		if np.array(box_center).shape == (1,): 
			box_center = box_center*3
			print("Only one dimension was provided for the box center! Assuming: ", box_center)
		elif np.array(box_center).shape == (2,) or np.array(box_center).shape[0] > 3: sys.exit("ERROR: Box center requires three dimensions ")
		box_center = np.array(box_center)
	elif (np.array(box_center).shape == (3,) ): box_center = np.array(box_center)
	
	print("-------------------------------------------------------------")
	
	
	# create star file object:
	datafile = startools.starfile(star_inp, verbosity=variables.v)
	
	
	# apply transformation
	new_transf = startools.apply_3D_coord_transform_to_ptcl_aln_params( \
		datafile.data_particles.data_array["_rlnAngleRot"] , \
		datafile.data_particles.data_array["_rlnAngleTilt"] , \
		datafile.data_particles.data_array["_rlnAnglePsi"] , \
		datafile.data_particles.data_array["_rlnOriginXAngst"] , \
		datafile.data_particles.data_array["_rlnOriginYAngst"] , \
		apix, \
		t , \
		euler , \
		box_center)
	
	# update datafile object:
	# structured array; fields have to be overwritten individually
	datafile.data_particles.data_array["_rlnAngleRot"  ] = new_transf[:,0]
	datafile.data_particles.data_array["_rlnAngleTilt" ] = new_transf[:,1]
	datafile.data_particles.data_array["_rlnAnglePsi"  ] = new_transf[:,2]
	datafile.data_particles.data_array["_rlnOriginXAngst"   ] = new_transf[:,3]
	datafile.data_particles.data_array["_rlnOriginYAngst"   ] = new_transf[:,4]
	
	
	
	
	
	# save datafile object:
	datafile.savestar(out_star)
	
	
	
if __name__ == "__main__": main()

