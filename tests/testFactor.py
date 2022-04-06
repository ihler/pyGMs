"""
testFactor.py

Unit tests for pyGMs factor class

Version 0.1.1 (2022-04-06)
(c) 2015 Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import unittest
import numpy as np
import sys
sys.path.append('../../')
from pyGMs import *

def eq_tol(F,G,tolerance):
	if (F.nvar != G.nvar) or (F.vars != G.vars):
		return False
	for x in range(F.numel()):
		if (not (np.isnan(F[x]) and np.isnan(G[x]))) and (np.abs(F[x]-G[x]) > tolerance):
			return False
	return True


# TODO:
#   argmax, argmin  (problem with ordering?)
#   marginal, etc   (problem with single-var args)
#




tol = 1e-5

class testFactor(unittest.TestCase):

	def setUp(self):
		return

	def testConstructor(self):
		x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
		F0 = Factor()
		self.assertEqual(F0.nvar , 0, msg='{} should be 0'.format(F0.nvar))
		self.assertEqual(F0.dims() , (1,))
		self.assertEqual(F0.numel() , 1  )
		self.assertEqual(F0.vars , VarSet())

		F1 = Factor( [x[1],x[2]] )	# 
		self.assertEqual(F1.nvar , 2, msg='{} should be 2'.format(F1.nvar))
		self.assertEqual(F1.dims() , (3,2))
		self.assertEqual(F1.numel() , 3*2 )
		self.assertEqual(F1.vars , VarSet([x[1],x[2]]))

		F2 = Factor( [x[1],x[2]] , 0.0 )
		self.assertEqual(F2.nvar , 2, msg='{} should be 2'.format(F2.nvar))
		self.assertEqual(F2.dims() , (3,2))
		self.assertEqual(F2.numel() , 3*2 )
		self.assertEqual(F2.vars , VarSet([x[1],x[2]]))

		F3 = Factor( [x[2],x[1]] , 0.0 )
		self.assertEqual(F3.nvar , 2, msg='{} should be 2'.format(F3.nvar))
		self.assertEqual(F3.dims() , (3,2))
		self.assertEqual(F3.numel() , 3*2 )
		self.assertEqual(F3.vars , VarSet([x[1],x[2]]))

		F4 = Factor( x[1] , [0.15,0.75,.1] )
		self.assertEqual(F4.nvar , 1, msg='{} should be 1'.format(F4.nvar))
		self.assertEqual(F4.dims() , (3,))
		self.assertEqual(F4.numel() , 3  )
		self.assertEqual(F4.vars , VarSet([x[1]]))

		F5 = Factor( [x[0],x[2]] , [ [0.1,0.9],[0.33,0.67] ] )
		self.assertEqual(F5.nvar , 2, msg='{} should be 2'.format(F5.nvar))
		self.assertEqual(F5.dims() , (2,2))
		self.assertEqual(F5.numel() , 2*2 )
		self.assertEqual(F5.vars , VarSet([x[0],x[2]]))

		F6 = Factor( [x[0],x[2]] , [  0.1,0.9 , 0.33,0.67  ] )
		self.assertEqual(F6.nvar , 2, msg='{} should be 2'.format(F6.nvar))
		self.assertEqual(F6.dims() , (2,2))
		self.assertEqual(F6.numel() , 2*2 )
		self.assertEqual(F6.vars , VarSet([x[0],x[2]]))

		F7 = Factor()		# repeat empty case to catch bad initializer errors
		self.assertEqual(F7.nvar , 0, msg='{} should be 0'.format(F7.nvar))
		self.assertEqual(F7.dims() , (1,))
		self.assertEqual(F7.numel() , 1  )
		self.assertEqual(F7.vars , VarSet())

		F8 = Factor( [x[0],x[4]] , [0.450,0.620,0.80,0.930,0.740,0.180,0.410,0.940,0.920,0.420] )
		self.assertEqual(F8.nvar , 2, msg='{} should be 2'.format(F8.nvar))
		self.assertEqual(F8.numel() , 2*5 )
		self.assertEqual( F8[(0,0)] , 0.45 )
		self.assertEqual( F8[(0,1)] , 0.80 )
		self.assertEqual( F8[(1,0)] , 0.62 )
		self.assertEqual( F8[(1,2)] , 0.18 )
		

	def testIndexing(self):
		x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
		F0 = Factor( [x[0],x[4]] , 2.0 )
		F1 = Factor( [x[0],x[4]] , 0.0 )
		for i in np.ndindex( F0.dims() ):
			F1[i] = 2.0
		self.assertTrue( eq_tol(F0,F1, tol) )
		F2 = Factor(  [x[0],x[4]] , 0.0 )
		for i in range(F0.numel()):
			F2[i] = 2.0
		self.assertTrue( eq_tol(F0,F2, tol) )


	#def testAssignment():

	def testCopy(self):
		x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
		F0 = Factor( [x[0],x[4]] , [0.450,0.620,0.80,0.930,0.740,0.180,0.410,0.940,0.920,0.420] )
		F1 = F0.copy()
		for i in np.ndindex( tuple(F0.dims()) ):
			F0[i] = 2.0
		self.assertEqual( F1[(0,0)] , 0.45 )
		self.assertEqual( F1[(0,1)] , 0.80 )
		self.assertEqual( F1[(1,0)] , 0.62 )
		self.assertEqual( F1[(1,2)] , 0.18 )

	#def testSwap(): (useful?)

	# TODO: binary checks	


	############ UNARY #####################################################
	def testUnaryAbs(self):
		x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
		F = Factor( [x[0],x[1]] , [-0.450,-0.620,0.80,0.930,-0.740,0.180] )
		r = Factor( [x[0],x[1]] , [ 0.450,0.620,0.80,0.930,0.740,0.180] )
		R = F.abs()
		self.assertEqual( r.nvar , R.nvar )
		self.assertEqual( r.vars , R.vars )
		self.assertTrue( eq_tol(r,R, tol) )

	def testUnaryExp(self):
		x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
		F = Factor( [x[0],x[1]] , [-0.450,-0.620,0.80,0.930,-0.740,0.180] )
		r = Factor( [x[0],x[1]] , [ 0.637628,0.537944,2.225541,2.534509,0.477114,1.197217 ] )
		R = F.exp()
		self.assertEqual( r.nvar , R.nvar )
		self.assertEqual( r.vars , R.vars )
		self.assertTrue( eq_tol(r,R, tol) )

	def testUnaryPower(self):
		x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
		F = Factor( [x[0],x[1]] , [0.450,0.620,0.80,0.930,0.740,0.180] )
		r = Factor( [x[0],x[1]] , [0.172611,0.349351,0.612066,0.852437,0.515596,0.022993 ] )

		p = 2.2;
		R = F.power(p)
		self.assertEqual( r.nvar , R.nvar )
		self.assertEqual( r.vars , R.vars )
		self.assertTrue( eq_tol(r,R, tol) )

	def testUnaryLog(self):
		x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
		F = Factor( [x[0],x[1]] , [  0.450, 0.620,0.80,1.930, 0.740,1.180 ] )
		r = Factor( [x[0],x[1]] , [ -0.798508,-0.478036,-0.223144,0.657520,-0.301105,0.165514  ] )
		R = F.log()
		self.assertEqual( r.nvar , R.nvar )
		self.assertEqual( r.vars , R.vars )
		self.assertTrue( eq_tol(r,R, tol) )

	def testUnaryLogBase2(self):
		x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
		F = Factor( [x[0],x[1]] , [  0.450, 0.620,0.80,1.930, 0.740,1.180 ] )
		r = Factor( [x[0],x[1]] , [ -1.152003,-0.689660,-0.321928,0.948601,-0.434403,0.238787 ] )
		R = F.log2()
		self.assertEqual( r.nvar , R.nvar )
		self.assertEqual( r.vars , R.vars )
		self.assertTrue( eq_tol(r,R, tol) )

	def testUnaryLogBase10(self):
		x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
		F = Factor( [x[0],x[1]] , [  0.450, 0.620,0.80,1.930, 0.740,1.180 ] )
		r = Factor( [x[0],x[1]] , [ -0.346787,-0.207608,-0.096910,0.285557,-0.130768,0.071882 ] )
		R = F.log10()
		self.assertEqual( r.nvar , R.nvar )
		self.assertEqual( r.vars , R.vars )
		self.assertTrue( eq_tol(r,R, tol) )


	############ ELIMINATION ##################################################
	def testElimSum2(self):
		x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
		F = Factor( [x[1],x[2]] , [0.150, 0.620, 0.20, 1.930, 0.740, 1.180] )
		r1 = Factor( [x[1]] , [ 2.0800, 1.3600, 1.3800 ] )
		R1 = F.sum( [x[2]] )
		self.assertEqual( r1.nvar , R1.nvar )
		self.assertEqual( r1.vars , R1.vars )
		self.assertTrue( eq_tol(r1,R1, tol) )

		r2 = Factor( [x[2]] , [ 0.9700, 3.8500 ] )
		R2 = F.sum( [x[1]] )
		self.assertEqual( r2.nvar , R2.nvar )
		self.assertEqual( r2.vars , R2.vars )
		self.assertTrue( eq_tol(r2,R2, tol) )

		R3 = F.marginal( [x[1]] )
		self.assertEqual( r1.nvar , R3.nvar )
		self.assertEqual( r1.vars , R3.vars )
		self.assertTrue( eq_tol(r1,R3, tol) )

		R4 = F.marginal( [x[2]] )
		self.assertEqual( r2.nvar , R4.nvar )
		self.assertEqual( r2.vars , R4.vars )
		self.assertTrue( eq_tol(r2,R4, tol) )

	def testElimSum3(self):
		x = [ Var(1,2), Var(3,3), Var(4,2) ]
		F = Factor( x , [0.8200,0.9100,0.1300,0.9200,0.6400,0.1000,0.2800,0.5500,0.9600,0.9700,0.1600,0.9800])
		# Check single-var marginals:
		r0 = Factor( [x[0]] , [ 2.9900,4.4300 ] )
		R0 = F.sum( [x[1],x[2]] )
		self.assertEqual( r0.nvar , R0.nvar )
		self.assertEqual( r0.vars , R0.vars )
		self.assertTrue( eq_tol(r0,R0, tol) )

		r1 = Factor( [x[1]] , [ 2.5600,2.9800,1.8800 ] )
		R1 = F.sum( [x[0],x[2]] )
		self.assertEqual( r1.nvar , R1.nvar )
		self.assertEqual( r1.vars , R1.vars )
		self.assertTrue( eq_tol(r1,R1, tol) )

		r2 = Factor( [x[2]] , [ 3.5200,3.9000 ] )
		R2 = F.sum( [x[0],x[1]] )
		self.assertEqual( r2.nvar , R2.nvar )
		self.assertEqual( r2.vars , R2.vars )
		self.assertTrue( eq_tol(r2,R2, tol) )

		# Next double-var marginals:
		r01 = Factor( [x[0],x[1]] , [ 1.1000,1.4600,1.0900,1.8900,0.8000,1.0800 ] )
		R01 = F.sum( [x[2]] )
		self.assertEqual( r01.nvar , R01.nvar )
		self.assertEqual( r01.vars , R01.vars )
		self.assertTrue( eq_tol(r01,R01, tol) )

		r02 = Factor( [x[0],x[2]] , [ 1.5900,1.9300,1.4000,2.5000 ] )
		R02 = F.sum( [x[1]] )
		self.assertEqual( r02.nvar , R02.nvar )
		self.assertEqual( r02.vars , R02.vars )
		self.assertTrue( eq_tol(r02,R02, tol) )

		r12 = Factor( [x[1],x[2]] , [ 1.7300,1.0500,0.7400,0.8300,1.9300,1.1400 ] )
		R12 = F.sum( [x[0]] )
		self.assertEqual( r12.nvar , R12.nvar )
		self.assertEqual( r12.vars , R12.vars )
		self.assertTrue( eq_tol(r12,R12, tol) )


	def testElimMax2(self):
		x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
		F = Factor( [x[1],x[2]] , [0.150, 0.620, 0.20, 1.930, 0.140, 1.180] )

		r1 = Factor( [x[1]] , [ 1.9300,0.6200,1.1800 ] )
		R1 = F.max( [x[2]] )
		self.assertEqual( r1.nvar , R1.nvar )
		self.assertEqual( r1.vars , R1.vars )
		self.assertTrue( eq_tol(r1,R1, tol) )

		r2 = Factor( [x[2]] , [ 0.6200, 1.9300 ] )
		R2 = F.max( [x[1]] )
		self.assertEqual( r2.nvar , R2.nvar )
		self.assertEqual( r2.vars , R2.vars )
		self.assertTrue( eq_tol(r2,R2, tol) )

		R3 = F.maxmarginal( [x[1]] )
		self.assertEqual( r1.nvar , R3.nvar )
		self.assertEqual( r1.vars , R3.vars )
		self.assertTrue( eq_tol(r1,R3, tol) )

		R4 = F.maxmarginal( [x[2]] )
		self.assertEqual( r2.nvar , R4.nvar )
		self.assertEqual( r2.vars , R4.vars )
		self.assertTrue( eq_tol(r2,R4, tol) )


	def testElimMax3(self):
		x = [ Var(1,2), Var(3,3), Var(4,2) ]
		F = Factor( x , [0.8200,0.9100,0.1300,0.9200,0.6400,0.1000,0.2800,0.5500,0.9600,0.9700,0.1600,0.9800])
		# Check single-var:
		r0 = Factor( [x[0]] , [ 0.9600,0.9800 ] )
		R0 = F.max( [x[1],x[2]] )
		self.assertEqual( r0.nvar , R0.nvar )
		self.assertEqual( r0.vars , R0.vars )
		self.assertTrue( eq_tol(r0,R0, tol) )

		r1 = Factor( [x[1]] , [ 0.9100,0.9700,0.9800 ] )
		R1 = F.max( [x[0],x[2]] )
		self.assertEqual( r1.nvar , R1.nvar )
		self.assertEqual( r1.vars , R1.vars )
		self.assertTrue( eq_tol(r1,R1, tol) )

		r2 = Factor( [x[2]] , [ 0.9200,0.9800 ] )
		R2 = F.max( [x[0],x[1]] )
		self.assertEqual( r2.nvar , R2.nvar )
		self.assertEqual( r2.vars , R2.vars )
		self.assertTrue( eq_tol(r2,R2, tol) )

		# Next double-var:
		r01 = Factor( [x[0],x[1]] , [ 0.8200,0.9100,0.9600,0.9700,0.6400,0.9800 ] )
		R01 = F.max( [x[2]] )
		self.assertEqual( r01.nvar , R01.nvar )
		self.assertEqual( r01.vars , R01.vars )
		self.assertTrue( eq_tol(r01,R01, tol) )

		r02 = Factor( [x[0],x[2]] , [ 0.8200,0.9200,0.9600,0.9800 ] )
		R02 = F.max( [x[1]] )
		self.assertEqual( r02.nvar , R02.nvar )
		self.assertEqual( r02.vars , R02.vars )
		self.assertTrue( eq_tol(r02,R02, tol) )

		r12 = Factor( [x[1],x[2]] , [ 0.9100,0.9200,0.6400,0.5500,0.9700,0.9800 ] )
		R12 = F.max( [x[0]] )
		self.assertEqual( r12.nvar , R12.nvar )
		self.assertEqual( r12.vars , R12.vars )
		self.assertTrue( eq_tol(r12,R12, tol) )


	def testElimMin2(self):
		x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
		F = Factor( [x[1],x[2]] , [0.150, 0.620, 0.20, 1.930, 0.140, 1.180] )

		r1 = Factor( [x[1]] , [ 0.1500,0.1400,0.2000 ] )
		R1 = F.min( [x[2]] )
		self.assertEqual( r1.nvar , R1.nvar )
		self.assertEqual( r1.vars , R1.vars )
		self.assertTrue( eq_tol(r1,R1, tol) )

		r2 = Factor( [x[2]] , [  0.1500,0.1400 ] )
		R2 = F.min( [x[1]] )
		self.assertEqual( r2.nvar , R2.nvar )
		self.assertEqual( r2.vars , R2.vars )
		self.assertTrue( eq_tol(r2,R2, tol) )

		R3 = F.minmarginal( [x[1]] )
		self.assertEqual( r1.nvar , R3.nvar )
		self.assertEqual( r1.vars , R3.vars )
		self.assertTrue( eq_tol(r1,R3, tol) )

		R4 = F.minmarginal( [x[2]] )
		self.assertEqual( r2.nvar , R4.nvar )
		self.assertEqual( r2.vars , R4.vars )
		self.assertTrue( eq_tol(r2,R4, tol) )


	def testElimMin3(self):
		x = [ Var(1,2), Var(3,3), Var(4,2) ]
		F = Factor( x , [0.8200,0.9100,0.1300,0.9200,0.6400,0.1000,0.2800,0.5500,0.9600,0.9700,0.1600,0.9800])
		# Check single-var:
		r0 = Factor( [x[0]] , [ 0.1300,0.1000 ] )
		R0 = F.min( [x[1],x[2]] )
		self.assertEqual( r0.nvar , R0.nvar )
		self.assertEqual( r0.vars , R0.vars )
		self.assertTrue( eq_tol(r0,R0, tol) )

		r1 = Factor( [x[1]] , [ 0.2800,0.1300,0.1000 ] )
		R1 = F.min( [x[0],x[2]] )
		self.assertEqual( r1.nvar , R1.nvar )
		self.assertEqual( r1.vars , R1.vars )
		self.assertTrue( eq_tol(r1,R1, tol) )

		r2 = Factor( [x[2]] , [ 0.1000,0.1600 ] )
		R2 = F.min( [x[0],x[1]] )
		self.assertEqual( r2.nvar , R2.nvar )
		self.assertEqual( r2.vars , R2.vars )
		self.assertTrue( eq_tol(r2,R2, tol) )

		# Next double-var:
		r01 = Factor( [x[0],x[1]] , [ 0.2800,0.5500,0.1300,0.9200,0.1600,0.1000 ] )
		R01 = F.min( [x[2]] )
		self.assertEqual( r01.nvar , R01.nvar )
		self.assertEqual( r01.vars , R01.vars )
		self.assertTrue( eq_tol(r01,R01, tol) )

		r02 = Factor( [x[0],x[2]] , [ 0.1300,0.1000,0.1600,0.5500 ] )
		R02 = F.min( [x[1]] )
		self.assertEqual( r02.nvar , R02.nvar )
		self.assertEqual( r02.vars , R02.vars )
		self.assertTrue( eq_tol(r02,R02, tol) )

		r12 = Factor( [x[1],x[2]] , [ 0.8200,0.1300,0.1000,0.2800,0.9600,0.1600 ] )
		R12 = F.min( [x[0]] )
		self.assertEqual( r12.nvar , R12.nvar )
		self.assertEqual( r12.vars , R12.vars )
		self.assertTrue( eq_tol(r12,R12, tol) )


	def testElimLse2(self):
		x = [ Var(0,2), Var(1,3), Var(2,2), Var(3,2), Var(4,5) ]
		F = Factor( [x[1],x[2]] , [ -1.89711998, -0.47803580, -1.60943791, 0.65752000, -0.30110509, 0.16551444 ] )

		r1 = Factor( [x[1]] , [ 0.73236789, 0.30748470, 0.3220835 ] )
		R1 = F.lse( [x[2]] )
		self.assertEqual( r1.nvar , R1.nvar )
		self.assertEqual( r1.vars , R1.vars )
		self.assertTrue( eq_tol(r1,R1, tol) )

		r2 = Factor( [x[2]] , [ -0.03045921, 1.34807315  ] )
		R2 = F.lse( [x[1]] )
		self.assertEqual( r2.nvar , R2.nvar )
		self.assertEqual( r2.vars , R2.vars )
		self.assertTrue( eq_tol(r2,R2, tol) )


	def testElimLse3(self):
		x = [ Var(1,2), Var(3,3), Var(4,2) ]
		F = Factor( x , [-0.19845094, -0.09431068, -2.04022083, -0.08338161, -0.44628710, -2.30258509, -1.27296568, -0.59783700, -0.04082199, -0.03045921, -1.83258146, -0.02020271])
		# Check single-var:
		r0 = Factor( [x[0]] , [ 1.09527339, 1.48839958 ] )
		R0 = F.lse( [x[1],x[2]] )
		self.assertEqual( r0.nvar , R0.nvar )
		self.assertEqual( r0.vars , R0.vars )
		self.assertTrue( eq_tol(r0,R0, tol) )

		r1 = Factor( [x[1]] , [ 0.94000726, 1.09192330, 0.63127178 ] )
		R1 = F.lse( [x[0],x[2]] )
		self.assertEqual( r1.nvar , R1.nvar )
		self.assertEqual( r1.vars , R1.vars )
		self.assertTrue( eq_tol(r1,R1, tol) )

		r2 = Factor( [x[2]] , [ 1.25846099, 1.36097655 ] )
		R2 = F.lse( [x[0],x[1]] )
		self.assertEqual( r2.nvar , R2.nvar )
		self.assertEqual( r2.vars , R2.vars )
		self.assertTrue( eq_tol(r2,R2, tol) )

		# Next double-var:
		r01 = Factor( [x[0],x[1]] , [ 0.09531018, 0.37843644, 0.08617770, 0.63657683, -0.22314355, 0.07696104 ] )
		R01 = F.lse( [x[2]] )
		self.assertEqual( r01.nvar , R01.nvar )
		self.assertEqual( r01.vars , R01.vars )
		self.assertTrue( eq_tol(r01,R01, tol) )

		r02 = Factor( [x[0],x[2]] , [ 0.46373402, 0.65752000, 0.33647224, 0.91629073 ] )
		R02 = F.lse( [x[1]] )
		self.assertEqual( r02.nvar , R02.nvar )
		self.assertEqual( r02.vars , R02.vars )
		self.assertTrue( eq_tol(r02,R02, tol) )

		r12 = Factor( [x[1],x[2]] , [ 0.54812141, 0.04879016, -0.30110509, -0.18632958, 0.65752000, 0.13102826 ] )
		R12 = F.lse( [x[0]] )
		self.assertEqual( r12.nvar , R12.nvar )
		self.assertEqual( r12.vars , R12.vars )
		self.assertTrue( eq_tol(r12,R12, tol) )

	
	# TODO: Power versions (sumPower, lsePower)


	############ BINARY OPERATORS #############################################

	def testArithmeticPlusScalar(self):
		x = [ Var(1,4) ]
		F = Factor( x , [ 0.220,0.520,0.390,0.490] )
		r = Factor( x , [  3.220,3.520,3.390,3.490] )
		self.assertTrue( eq_tol(r, F+3.0, tol) )
		self.assertTrue( eq_tol(r, 3.0+F, tol) )
		F += 3.0
		self.assertTrue( eq_tol(r, F, tol) )

	def testArithmeticPlusSameScope(self):
		x = [ Var(1,3), Var(3,3) ]
		a = Factor( x, [ 0.8200,0.9100,0.1300,0.1900,0.6400,0.1000,0.2800,0.5500,0.9600 ] )
		b = Factor( x, [ 0.4500,0.6200,0.8000,0.9300,0.7400,0.1800,0.4100,0.9400,0.9200 ] )
		r = Factor( x, [  1.270,1.530, 0.930, 1.120, 1.380, 0.280, 0.690, 1.490, 1.880 ] )
		self.assertTrue( eq_tol(r, a+b, tol) )
		self.assertTrue( eq_tol(r, b+a, tol) )
		a += b
		self.assertTrue( eq_tol(r, a, tol) )

	def testArithmeticPlusDisjointScope(self):
		x = [ Var(1,4), Var(2,3), Var(4,3) ]
		a = Factor( [x[0]], [ 0.220,0.520,0.390,0.490] )
		b = Factor( [x[1],x[2]], [ 0.460,0.690,0.210,0.310,0.210,0.840,0.020,0.820,0.910] )
		r = Factor( x, [ 0.680,0.980,0.850,0.950,0.910,1.210,1.080,1.180,0.430,0.730,0.60,0.70,0.530,0.830,0.70,0.80,0.430,0.730,0.60,0.70,1.060,1.360,1.230,1.330,0.240,0.540,0.410,0.510,1.040,1.340,1.210,1.310,1.130,1.430,1.30,1.40] )
		self.assertTrue( eq_tol(r, a+b, tol) )
		self.assertTrue( eq_tol(r, b+a, tol) )
		a += b
		self.assertTrue( eq_tol(r, a, tol) )
		b += Factor( [x[0]], [ 0.220,0.520,0.390,0.490] )
		self.assertTrue( eq_tol(r, b, tol) )

	def testArithmeticPlusOverlap2(self):
		x = [ Var(1,2), Var(3,3) ]
		a = Factor( [x[0]], [ 0.340,0.420 ] )
		b = Factor( [x[0],x[1]], [ 0.010,0.890,0.520,0.550,0.760,0.140 ] )
		r = Factor( x, [  0.350, 1.310, 0.860, 0.970, 1.100, 0.560 ] )
		self.assertTrue( eq_tol(r, a+b, tol) )
		self.assertTrue( eq_tol(r, b+a, tol) )
		c = Factor( [x[1]], [ 0.340,0.420,0.990] )
		d = Factor( [x[0],x[1]], [ 0.010,0.890,0.520,0.550,0.760,0.140] )
		r2 = Factor( x, [  0.350, 1.230, 0.940, 0.970, 1.750, 1.130 ] )
		self.assertTrue( eq_tol(r2, c+d, tol) )
		self.assertTrue( eq_tol(r2, d+c, tol) )
		# Now in-place versions
		a2,b2 = Factor(a.v,a.t),Factor(b.v,b.t)
		a2 += b
		b2 += a
		self.assertTrue( eq_tol(r, a2, tol) )
		self.assertTrue( eq_tol(r, b2, tol) )
		c2,d2 = Factor(c.v,c.t),Factor(d.v,d.t)
		c2 += d
		d2 += c
		self.assertTrue( eq_tol(r2, c2, tol) )
		self.assertTrue( eq_tol(r2, d2, tol) )

	def testArithmeticPlusOverlap3(self):
		x = [ Var(1,2), Var(3,3), Var(4,2) ]
		a = Factor( [x[0],x[2]], [ 0.450,0.620,0.800,0.930 ] )
		b = Factor( [x[0],x[1]], [ 0.010,0.890,0.520,0.550,0.760,0.140 ] )
		r = Factor( x, [  0.460, 1.51, 0.970, 1.170, 1.210, 0.760, 0.810, 1.820, 1.320, 1.480, 1.560, 1.070 ] )
		self.assertTrue( eq_tol(r, a+b, tol) )
		self.assertTrue( eq_tol(r, b+a, tol) )
		c = Factor( [x[0],x[2]], [ 0.450,0.620,0.800,0.930 ] )
		d = Factor( [x[1],x[2]], [ 0.010,0.890,0.520,0.550,0.760,0.140] )
		r2 = Factor( x, [  0.460, 0.630, 1.340, 1.51, 0.970, 1.140, 1.350, 1.480, 1.560, 1.690, 0.940, 1.070 ] )
		self.assertTrue( eq_tol(r2, c+d, tol) )
		self.assertTrue( eq_tol(r2, d+c, tol) )
		# Now in-place versions
		a2,b2 = Factor(a.v,a.t),Factor(b.v,b.t)
		a2 += b
		b2 += a
		self.assertTrue( eq_tol(r, a2, tol) )
		self.assertTrue( eq_tol(r, b2, tol) )
		c2,d2 = Factor(c.v,c.t),Factor(d.v,d.t)
		c2 += d
		d2 += c
		self.assertTrue( eq_tol(r2, c2, tol) )
		self.assertTrue( eq_tol(r2, d2, tol) )


	def testArithmeticMinusScalar(self):
		x = [ Var(1,4) ]
		F = Factor( x , [ 0.220,0.520,0.390,0.490 ] )
		r = Factor( x , [  -2.780, -2.480, -2.610, -2.510 ] )
		rn= Factor( x , [   2.780,  2.480,  2.610,  2.510 ] )
		self.assertTrue( eq_tol(r , F-3.0, tol) )
		self.assertTrue( eq_tol(rn, 3.0-F, tol) )
		F -= 3.0
		self.assertTrue( eq_tol(r, F, tol) )

	def testArithmeticMinusSameScope(self):
		x = [ Var(1,3), Var(3,3) ]
		a = Factor( x, [ 0.8200,0.9100,0.1300,0.1900,0.6400,0.1000,0.2800,0.5500,0.9600 ] )
		b = Factor( x, [ 0.4500,0.6200,0.8000,0.9300,0.7400,0.1800,0.4100,0.9400,0.9200 ] )
		r = Factor( x, [  0.370, 0.290, -0.670, -0.740, -0.100, -0.080, -0.130, -0.390, 0.040 ] )
		rn= Factor( x, [ -0.370,-0.290,  0.670,  0.740,  0.100,  0.080,  0.130,  0.390,-0.040 ] )
		self.assertTrue( eq_tol(r , a-b, tol) )
		self.assertTrue( eq_tol(rn, b-a, tol) )
		a -= b
		self.assertTrue( eq_tol(r, a, tol) )

	def testArithmeticMinusDisjointScope(self):
		x = [ Var(1,4), Var(2,3), Var(4,3) ]
		a = Factor( [x[0]], [ 0.220,0.520,0.390,0.490 ] )
		b = Factor( [x[1],x[2]], [ 0.460,0.690,0.210,0.310,0.210,0.840,0.020,0.820,0.910 ] )
		r = Factor( x, [  -0.240, 0.060, -0.070, 0.030, -0.470, -0.170, -0.300, -0.200, 0.010, 0.310, 0.180, 0.280, -0.090, 0.210, 0.080, 0.180, 0.010, 0.310, 0.180, 0.280, -0.620, -0.320, -0.450, -0.350, 0.200, 0.500, 0.370, 0.470, -0.600, -0.300, -0.430, -0.330, -0.690, -0.390, -0.520, -0.420 ] )
		rn= Factor( x, [   0.240,-0.060,  0.070,-0.030,  0.470,  0.170,  0.300,  0.200,-0.010,-0.310,-0.180,-0.280,  0.090,-0.210,-0.080,-0.180,-0.010,-0.310,-0.180,-0.280,  0.620,  0.320,  0.450,  0.350,-0.200,-0.500,-0.370,-0.470,  0.600,  0.300,  0.430,  0.330,  0.690,  0.390,  0.520,  0.420 ] )
		self.assertTrue( eq_tol(r , a-b, tol) )
		self.assertTrue( eq_tol(rn, b-a, tol) )
		a -= b
		self.assertTrue( eq_tol(r, a, tol) )
		b -= Factor( [x[0]], [ 0.220,0.520,0.390,0.490] )
		self.assertTrue( eq_tol(rn, b, tol) )

	def testArithmeticMinusOverlap2(self):
		x = [ Var(1,2), Var(3,3) ]
		a = Factor( [x[0]], [ 0.340,0.420 ] )
		b = Factor( [x[0],x[1]], [ 0.010,0.890,0.520,0.550,0.760,0.140 ] )
		r = Factor( x, [  0.330, -0.470, -0.180, -0.130, -0.420, 0.280 ] )
		rn= Factor( x, [ -0.330,  0.470,  0.180,  0.130,  0.420,-0.280 ] )
		self.assertTrue( eq_tol(r , a-b, tol) )
		self.assertTrue( eq_tol(rn, b-a, tol) )
		c = Factor( [x[1]], [ 0.340,0.420,0.990] )
		d = Factor( [x[0],x[1]], [ 0.010,0.890,0.520,0.550,0.760,0.140] )
		r2 = Factor( x, [  0.330, -0.550, -0.100, -0.130, 0.230, 0.850 ] )
		r2n= Factor( x, [ -0.330,  0.550,  0.100,  0.130,-0.230,-0.850 ] )
		self.assertTrue( eq_tol(r2 , c-d, tol) )
		self.assertTrue( eq_tol(r2n, d-c, tol) )
		# Now in-place versions
		a2,b2 = Factor(a.v,a.t),Factor(b.v,b.t)
		a2 -= b
		b2 -= a
		self.assertTrue( eq_tol(r, a2, tol) )
		self.assertTrue( eq_tol(rn, b2, tol) )
		c2,d2 = Factor(c.v,c.t),Factor(d.v,d.t)
		c2 -= d
		d2 -= c
		self.assertTrue( eq_tol(r2, c2, tol) )
		self.assertTrue( eq_tol(r2n, d2, tol) )


	def testArithmeticMinusOverlap3(self):
		x = [ Var(1,2), Var(3,3), Var(4,2) ]
		a = Factor( [x[0],x[2]], [ 0.450,0.620,0.800,0.930 ] )
		b = Factor( [x[0],x[1]], [ 0.010,0.890,0.520,0.550,0.760,0.140 ] )
		r = Factor( x, [  0.440, -0.270, -0.070, 0.070, -0.310, 0.480, 0.790, 0.040, 0.280, 0.380, 0.040, 0.790 ] )
		rn= Factor( x, [ -0.440,  0.270,  0.070,-0.070,  0.310,-0.480,-0.790,-0.040,-0.280,-0.380,-0.040,-0.790 ] )
		self.assertTrue( eq_tol(r , a-b, tol) )
		self.assertTrue( eq_tol(rn, b-a, tol) )
		c = Factor( [x[0],x[2]], [ 0.450,0.620,0.800,0.930 ] )
		d = Factor( [x[1],x[2]], [ 0.010,0.890,0.520,0.550,0.760,0.140] )
		r2 = Factor( x, [  0.440, 0.610, -0.440, -0.270, -0.070, 0.100, 0.250, 0.380, 0.040, 0.170, 0.660, 0.790 ] )
		r2n= Factor( x, [ -0.440,-0.610,  0.440,  0.270,  0.070,-0.100,-0.250,-0.380,-0.040,-0.170,-0.660,-0.790 ] )
		self.assertTrue( eq_tol(r2 , c-d, tol) )
		self.assertTrue( eq_tol(r2n, d-c, tol) )
		# Now in-place versions
		a2,b2 = Factor(a.v,a.t),Factor(b.v,b.t)
		a2 -= b
		b2 -= a
		self.assertTrue( eq_tol(r, a2, tol) )
		self.assertTrue( eq_tol(rn, b2, tol) )
		c2,d2 = Factor(c.v,c.t),Factor(d.v,d.t)
		c2 -= d
		d2 -= c
		self.assertTrue( eq_tol(r2, c2, tol) )
		self.assertTrue( eq_tol(r2n, d2, tol) )


	def testArithmeticTimesScalar(self):
		x = [ Var(1,4) ]
		F = Factor( x , [ 0.220,0.520,0.390,0.490 ] )
		r = Factor( x , [  0.6600000, 1.5600000, 1.1700000, 1.4700000 ] )
		self.assertTrue( eq_tol(r, F*3.0, tol) )
		self.assertTrue( eq_tol(r, 3.0*F, tol) )
		F *= 3.0
		self.assertTrue( eq_tol(r, F, tol) )

	def testArithmeticTimesSameScope(self):
		x = [ Var(1,3), Var(3,3) ]
		a = Factor( x, [ 0.8200,0.9100,0.1300,0.1900,0.6400,0.1000,0.2800,0.5500,0.9600 ] )
		b = Factor( x, [ 0.4500,0.6200,0.8000,0.9300,0.7400,0.1800,0.4100,0.9400,0.9200 ] )
		r = Factor( x, [  0.3690000, 0.5642000, 0.1040000, 0.1767000, 0.4736000, 0.0180000, 0.1148000, 0.5170000, 0.8832000 ] )
		self.assertTrue( eq_tol(r, a*b, tol) )
		self.assertTrue( eq_tol(r, b*a, tol) )
		a *= b
		self.assertTrue( eq_tol(r, a, tol) )

	def testArithmeticTimesDisjointScope(self):
		x = [ Var(1,4), Var(2,3), Var(4,3) ]
		a = Factor( [x[0]], [ 0.220,0.520,0.390,0.490 ] )
		b = Factor( [x[1],x[2]], [ 0.460,0.690,0.210,0.310,0.210,0.840,0.020,0.820,0.910 ] )
		r = Factor( x, [  0.1012000, 0.2392000, 0.1794000, 0.2254000, 0.1518000, 0.3588000, 0.2691000, 0.3381000, 0.0462000, 0.1092000, 0.0819000, 0.1029000, 0.0682000, 0.1612000, 0.1209000, 0.1519000, 0.0462000, 0.1092000, 0.0819000, 0.1029000, 0.1848000, 0.4368000, 0.3276000, 0.4116000, 0.0044000, 0.0104000, 0.0078000, 0.0098000, 0.1804000, 0.4264000, 0.3198000, 0.4018000, 0.2002000, 0.4732000, 0.3549000, 0.4459000 ] )
		self.assertTrue( eq_tol(r, a*b, tol) )
		self.assertTrue( eq_tol(r, b*a, tol) )
		a *= b
		self.assertTrue( eq_tol(r, a, tol) )
		b *= Factor( [x[0]], [ 0.220,0.520,0.390,0.490] )
		self.assertTrue( eq_tol(r, b, tol) )

	def testArithmeticTimesOverlap2(self):
		x = [ Var(1,2), Var(3,3) ]
		a = Factor( [x[0]], [ 0.340,0.420 ] )
		b = Factor( [x[0],x[1]], [ 0.010,0.890,0.520,0.550,0.760,0.140 ] )
		r = Factor( x, [   0.0034000, 0.3738000, 0.1768000, 0.2310000, 0.2584000, 0.0588000 ] )
		self.assertTrue( eq_tol(r, a*b, tol) )
		self.assertTrue( eq_tol(r, b*a, tol) )
		c = Factor( [x[1]], [ 0.340,0.420,0.990] )
		d = Factor( [x[0],x[1]], [ 0.010,0.890,0.520,0.550,0.760,0.140] )
		r2 = Factor( x, [   0.0034000, 0.3026000, 0.2184000, 0.2310000, 0.7524000, 0.1386000 ] )
		self.assertTrue( eq_tol(r2, c*d, tol) )
		self.assertTrue( eq_tol(r2, d*c, tol) )
		# Now in-place versions
		a2,b2 = Factor(a.v,a.t),Factor(b.v,b.t)
		a2 *= b
		b2 *= a
		self.assertTrue( eq_tol(r, a2, tol) )
		self.assertTrue( eq_tol(r, b2, tol) )
		c2,d2 = Factor(c.v,c.t),Factor(d.v,d.t)
		c2 *= d
		d2 *= c
		self.assertTrue( eq_tol(r2, c2, tol) )
		self.assertTrue( eq_tol(r2, d2, tol) )

	def testArithmeticTimesOverlap3(self):
		x = [ Var(1,2), Var(3,3), Var(4,2) ]
		a = Factor( [x[0],x[2]], [ 0.450,0.620,0.800,0.930 ] )
		b = Factor( [x[0],x[1]], [ 0.010,0.890,0.520,0.550,0.760,0.140 ] )
		r = Factor( x, [   0.0045000, 0.5518000, 0.2340000, 0.3410000, 0.3420000, 0.0868000, 0.0080000, 0.8277000, 0.4160000, 0.5115000, 0.6080000, 0.1302000 ] )
		self.assertTrue( eq_tol(r, a*b, tol) )
		self.assertTrue( eq_tol(r, b*a, tol) )
		c = Factor( [x[0],x[2]], [ 0.450,0.620,0.800,0.930 ] )
		d = Factor( [x[1],x[2]], [ 0.010,0.890,0.520,0.550,0.760,0.140] )
		r2 = Factor( x, [   0.0045000, 0.0062000, 0.4005000, 0.5518000, 0.2340000, 0.3224000, 0.4400000, 0.5115000, 0.6080000, 0.7068000, 0.1120000, 0.1302000 ] )
		self.assertTrue( eq_tol(r2, c*d, tol) )
		self.assertTrue( eq_tol(r2, d*c, tol) )
		# Now in-place versions
		a2,b2 = Factor(a.v,a.t),Factor(b.v,b.t)
		a2 *= b
		b2 *= a
		self.assertTrue( eq_tol(r, a2, tol) )
		self.assertTrue( eq_tol(r, b2, tol) )
		c2,d2 = Factor(c.v,c.t),Factor(d.v,d.t)
		c2 *= d
		d2 *= c
		self.assertTrue( eq_tol(r2, c2, tol) )
		self.assertTrue( eq_tol(r2, d2, tol) )


	def testArithmeticDivideScalar(self):
		x = [ Var(1,4) ]
		F = Factor( x , [ 0.220,0.520,0.390,0.490 ] )
		r = Factor( x , [ 0.0733333, 0.1733333, 0.1300000, 0.1633333 ] )
		rn= Factor( x , [13.6363637, 5.7692307, 7.6923076, 6.1224489 ] )
		self.assertTrue( eq_tol(r , F/3.0, tol) )
		self.assertTrue( eq_tol(rn, 3.0/F, tol) )
		F /= 3.0
		self.assertTrue( eq_tol(r, F, tol) )

	def testArithmeticDivideSameScope(self):
		x = [ Var(1,3), Var(3,3) ]
		a = Factor( x, [ 0.8200,0.9100,0.1300,0.1900,0.6400,0.1000,0.2800,0.5500,0.9600 ] )
		b = Factor( x, [ 0.4500,0.6200,0.8000,0.9300,0.7400,0.1800,0.4100,0.9400,0.9200 ] )
		r = Factor( x, [  1.8222222, 1.4677419, 0.1625000, 0.2043010, 0.8648648, 0.5555555, 0.6829268, 0.5851063, 1.0434782 ] )
		rn= Factor( x, [  0.5487804, 0.6813186, 6.1538461, 4.8947368, 1.1562500, 1.8000000, 1.4642857, 1.7090909, 0.9583333 ] )
		self.assertTrue( eq_tol(r , a/b, tol) )
		self.assertTrue( eq_tol(rn, b/a, tol) )
		a /= b
		self.assertTrue( eq_tol(r, a, tol) )

	def testArithmeticDivideDisjointScope(self):
		x = [ Var(1,4), Var(2,3), Var(4,3) ]
		a = Factor( [x[0]], [ 0.220,0.520,0.390,0.490 ] )
		b = Factor( [x[1],x[2]], [ 0.460,0.690,0.210,0.310,0.210,0.840,0.020,0.820,0.910 ] )
		r = Factor( x, [  0.4782608, 1.1304347, 0.8478260, 1.0652174, 0.3188405, 0.7536231, 0.5652174, 0.7101449, 1.0476190, 2.4761904, 1.8571428, 2.3333333, 0.7096774, 1.6774193, 1.2580645, 1.5806451, 1.0476190, 2.4761904, 1.8571428, 2.3333333, 0.2619047, 0.6190476, 0.4642857, 0.5833333, 11.0000000, 26.0000000, 19.5000000, 24.5000000, 0.2682926, 0.6341463, 0.4756097, 0.5975609, 0.2417582, 0.5714285, 0.4285714, 0.5384615 ] )
		rn= Factor( x, [  2.0909091, 0.8846153, 1.1794871, 0.9387755, 3.1363636, 1.3269230, 1.7692307, 1.4081632, 0.9545454, 0.4038461, 0.5384615, 0.4285714, 1.4090909, 0.5961538, 0.7948718, 0.6326530, 0.9545454, 0.4038461, 0.5384615, 0.4285714, 3.8181818, 1.6153846, 2.1538461, 1.7142857, 0.0909091, 0.0384615, 0.0512820, 0.0408163, 3.7272727, 1.5769230, 2.1025641, 1.6734693, 4.1363636, 1.7500000, 2.3333333, 1.8571428 ] )
		self.assertTrue( eq_tol(r , a/b, tol) )
		self.assertTrue( eq_tol(rn, b/a, tol) )
		a /= b
		self.assertTrue( eq_tol(r, a, tol) )
		b /= Factor( [x[0]], [ 0.220,0.520,0.390,0.490] )
		self.assertTrue( eq_tol(rn, b, tol) )

	def testArithmeticDivideOverlap2(self):
		x = [ Var(1,2), Var(3,3) ]
		a = Factor( [x[0]], [ 0.340,0.420 ] )
		b = Factor( [x[0],x[1]], [ 0.010,0.890,0.520,0.550,0.760,0.140 ] )
		r = Factor( x, [ 34, 0.4719101, 0.6538461, 0.7636363, 0.4473684, 3.0000000 ] )
		rn= Factor( x, [ 0.02941176, 2.1190476, 1.5294117, 1.3095238, 2.2352941, 0.3333333] )
		self.assertTrue( eq_tol(r , a/b, tol) )
		self.assertTrue( eq_tol(rn, b/a, tol) )
		c = Factor( [x[1]], [ 0.340,0.420,0.990] )
		d = Factor( [x[0],x[1]], [ 0.010,0.890,0.520,0.550,0.760,0.140] )
		r2 = Factor( x, [   34.0000000, 0.3820224, 0.8076923, 0.7636363, 1.3026315, 7.0714285 ] )
		r2n= Factor( x, [   0.0294117, 2.6176470, 1.2380952, 1.3095238, 0.7676767, 0.1414141 ] )
		self.assertTrue( eq_tol(r2 , c/d, tol) )
		self.assertTrue( eq_tol(r2n, d/c, tol) )
		# Now in-place versions
		a2,b2 = Factor(a.v,a.t),Factor(b.v,b.t)
		a2 /= b
		b2 /= a
		self.assertTrue( eq_tol(r, a2, tol) )
		self.assertTrue( eq_tol(rn, b2, tol) )
		c2,d2 = Factor(c.v,c.t),Factor(d.v,d.t)
		c2 /= d
		d2 /= c
		self.assertTrue( eq_tol(r2, c2, tol) )
		self.assertTrue( eq_tol(r2n, d2, tol) )

	def testArithmeticDivideOverlap3(self):
		x = [ Var(1,2), Var(3,3), Var(4,2) ]
		a = Factor( [x[0],x[2]], [ 0.450,0.620,0.800,0.930 ] )
		b = Factor( [x[0],x[1]], [ 0.010,0.890,0.520,0.550,0.760,0.140 ] )
		r = Factor( x, [ 45.0000000, 0.6966292, 0.8653846, 1.1272727, 0.5921052, 4.4285714, 80.0000000, 1.0449438, 1.5384615, 1.6909091, 1.0526315, 6.6428571 ] )
		rn= Factor( x, [  0.0222222, 1.4354838, 1.1555555, 0.8870967, 1.6888888, 0.22580646, 0.01250000, 0.9569892, 0.6500000, 0.5913978, 0.9500000, 0.1505376 ] )
		self.assertTrue( eq_tol(r , a/b, tol) )
		self.assertTrue( eq_tol(rn, b/a, tol) )
		c = Factor( [x[0],x[2]], [ 0.450,0.620,0.800,0.930 ] )
		d = Factor( [x[1],x[2]], [ 0.010,0.890,0.520,0.550,0.760,0.140] )
		r2 = Factor( x, [ 45.0000000, 62.0000000, 0.5056179, 0.6966292, 0.8653846, 1.1923077, 1.4545454, 1.6909091, 1.0526315, 1.2236842, 5.7142857, 6.6428571 ] )
		r2n= Factor( x, [  0.0222222, 0.0161290, 1.9777777, 1.4354838, 1.1555555, 0.8387096, 0.6875000, 0.5913978, 0.9500000, 0.8172043, 0.1750000, 0.1505376 ] )
		self.assertTrue( eq_tol(r2 , c/d, tol) )
		self.assertTrue( eq_tol(r2n, d/c, tol) )
		# Now in-place versions
		a2,b2 = Factor(a.v,a.t),Factor(b.v,b.t)
		a2 /= b
		b2 /= a
		self.assertTrue( eq_tol(r, a2, tol) )
		self.assertTrue( eq_tol(rn, b2, tol) )
		c2,d2 = Factor(c.v,c.t),Factor(d.v,d.t)
		c2 /= d
		d2 /= c
		self.assertTrue( eq_tol(r2, c2, tol) )
		self.assertTrue( eq_tol(r2n, d2, tol) )

	def testArgmax(self):
		x = [ Var(1,4), Var(2,3), Var(4,3) ]
		a = Factor( [x[0]], [ 0.220,0.520,0.390,0.490 ] )
		b = Factor( [x[1],x[2]], [ 0.460,0.690,0.210,0.310,0.210,0.840,0.020,0.820,0.910 ] )
		r = Factor( x, [  0.4782608, 1.1304347, 0.8478260, 1.0652174, 0.3188405, 0.7536231, 0.5652174, 0.7101449, 1.0476190, 2.4761904, 1.8571428, 2.3333333, 0.7096774, 1.6774193, 1.2580645, 1.5806451, 1.0476190, 2.4761904, 1.8571428, 2.3333333, 0.2619047, 0.6190476, 0.4642857, 0.5833333, 11.0000000, 26.0000000, 19.5000000, 24.5000000, 0.2682926, 0.6341463, 0.4756097, 0.5975609, 0.2417582, 0.5714285, 0.4285714, 0.5384615 ] )
		self.assertEqual( a.argmax() , (1,) )
		self.assertEqual( b.argmax() , (2,2) )
		self.assertEqual( r.argmax() , (1,0,2) )
		self.assertEqual( (a/b).argmax() , (1,0,2) )


if __name__ == '__main__':
	unittest.main()



