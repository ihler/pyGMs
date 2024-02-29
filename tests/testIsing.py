"""
testIsing.py

Unit tests for pyGMs Ising class (specialty Ising / Boltzmann models)

Version 0.1.1 (2022-04-06)
(c) 2021- Alexander Ihler under the FreeBSD license; see license.txt for details.
"""

import unittest
import numpy as np
import sys
sys.path.append('../../')
import pyGMs as gm
import pyGMs.ising

import matplotlib
import matplotlib.pyplot as plt
import copy


def close(a,b, tol=1e-4): return np.max(np.abs(a-b))<tol

X = [gm.Var(i,2) for i in range(6)]      # default variables for later use

# Observation data set for later use
D = np.array([[0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
              [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
              [0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
              [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0]]).T

# Some explicit model parameters
model_params = [
         ([ 0.21, -0.41,  0.86, -0.22, -0.10 , -0.20],
         {(0,1): -.62, (1,2): -.41, (0,3): .14, (1,4): -1.0, (3,4): .60, (3,5): .26, (4,5): -.80 } 
        )
        ]

class testIsing(unittest.TestCase):

    def setUp(self): 
        return

    def testConstructorFromFactors(self):
        for th1,th2 in model_params:
            F1 = [gm.Factor([X[i]],[0,th1[i]]).exp() for i in range(len(th1))]
            F2 = [gm.Factor([X[i],X[j]],[[0,0],[0,th2[i,j]]]).exp() for i,j in th2]
            model = gm.ising.Ising(F1+F2)

    def testBasicAccess(self):
        th1,th2 = model_params[0]
        F1 = [gm.Factor([X[i]],[0,th1[i]]).exp() for i in range(len(th1))]
        F2 = [gm.Factor([X[i],X[j]],[[0,0],[0,th2[i,j]]]).exp() for i,j in th2]
        model = gm.ising.Ising(F1+F2)
        #
        self.assertEqual(model.nvar , 6)
        self.assertEqual( len(model.vars) , 6 )
        self.assertEqual(model.nfactors , 2*7)          # not doubled?
        self.assertEqual(len(model.factors) , (1+6+7))  # weird to not match nfactors?
        self.assertEqual(model.isBinary(), True)
        self.assertEqual(model.isPairwise(), True)

    def testConnectivity(self):
        th1,th2 = model_params[0]
        F1 = [gm.Factor([X[i]],[0,th1[i]]).exp() for i in range(len(th1))]
        F2 = [gm.Factor([X[i],X[j]],[[0,0],[0,th2[i,j]]]).exp() for i,j in th2]
        model = gm.ising.Ising(F1+F2)
        #
        self.assertEqual(len(model.factorsWith(1)) , 4)
        self.assertEqual(len(model.factorsWithAny([3,4])) , (2+5))
        self.assertEqual(model.degree(1) , 3)
        self.assertEqual(len(model.markovBlanket(1)) , 3)
        self.assertEqual(len(model.markovBlanket(5)) , 2)

    def testJoint(self):
        th1,th2 = model_params[0]
        F1 = [gm.Factor([X[i]],[0,th1[i]]).exp() for i in range(len(th1))]
        F2 = [gm.Factor([X[i],X[j]],[[0,0],[0,th2[i,j]]]).exp() for i,j in th2]
        model = gm.ising.Ising(F1+F2)
        fgraph= gm.GraphModel(F1+F2)
        #
        self.assertTrue( close(model.joint().t, fgraph.joint().t, 1e-4) )




######## DATA #############################

    def testLogValue(self):
        th1,th2 = model_params[0]
        F1 = [gm.Factor([X[i]],[0,th1[i]]).exp() for i in range(len(th1))]
        F2 = [gm.Factor([X[i],X[j]],[[0,0],[0,th2[i,j]]]).exp() for i,j in th2]
        model = gm.ising.Ising(F1+F2)
        fgraph= gm.GraphModel(F1+F2)
        fgph2 = gm.GraphModel(model.factors)
        #
        lvT= np.array([-0.57,0.01,1.14,-0.22,0.87,0.63,0.86,-1.4,-0.12,0.7,0.28,0.87,0.75,0.97, 
             -2.06,1.05,0.66,0.75,-0.24,1.05,-0.22,-0.57,0.87,1.05,-0.18,-0.16,0.99,-0.45,0.97,-0.18])
        lv = fgraph.logValue(D); self.assertTrue( np.max(np.abs(lv-lvT)) < 1e-4 )  ## TODO: FIX: Transpose
        lv = fgph2.logValue(D);  self.assertTrue( np.max(np.abs(lv-lvT)) < 1e-4 )  ## TODO: FIX same
        lv = model.logValue(D);  self.assertTrue( np.max(np.abs(lv-lvT)) < 1e-4 )

    def testPseudolikelihood(self):
        th1,th2 = model_params[0]
        F1 = [gm.Factor([X[i]],[0,th1[i]]).exp() for i in range(len(th1))]
        F2 = [gm.Factor([X[i],X[j]],[[0,0],[0,th2[i,j]]]).exp() for i,j in th2]
        model = gm.ising.Ising(F1+F2)
        fgraph= gm.GraphModel(F1+F2)
        #
        plT=np.array([-4.01, -4.04, -2.77, -5.11, -3.09, -3.2 , -3.35, -5.76, -3.82,
             -3.49, -3.7 , -3.09, -3.33, -3.07, -7.39, -2.92, -3.37, -3.33,
             -4.92, -2.92, -5.11, -4.01, -3.09, -2.92, -4.25, -3.84, -3.53,
             -4.91, -3.07, -4.25])
        pl = gm.pseudologlikelihood(fgraph, D); self.assertTrue( close(pl,plT, 2e-2) )
        pl = model.pseudolikelihood(D);         self.assertTrue( close(pl,plT, 2e-2) )

    def testConnectedComp(self):
        th1,th2 = model_params[0]
        F1 = [gm.Factor([X[i]],[0,th1[i]]).exp() for i in range(len(th1))]
        F2 = [gm.Factor([X[i],X[j]],[[0,0],[0,th2[i,j]]]).exp() for i,j in th2]
        ccmod = gm.ising.Ising(F1);                self.assertEqual( len(ccmod.connectedComponents()) , 6 )
        ccmod = gm.ising.Ising(F1+F2[:2]+F2[-1:]); self.assertEqual( len(ccmod.connectedComponents()) , 3 )
        ccmod = gm.ising.Ising(F1+F2);             self.assertEqual( len(ccmod.connectedComponents()) , 1 )


    # Check conditioning model on a few variables


    # Draw samples and evaluate mean log value ("close" to expected?)

    #### Approximate Inference  ##############################

    def testLoopyBP(self):
        th1,th2 = model_params[0]
        F1 = [gm.Factor([X[i]],[0,th1[i]]).exp() for i in range(len(th1))]
        F2 = [gm.Factor([X[i],X[j]],[[0,0],[0,th2[i,j]]]).exp() for i,j in th2]
        model = gm.ising.Ising(F1+F2);
        lnZ, bel = gm.ising.LBP(model)


    #### Parameter fitting  ##################################

    def testRefitPLL(self):
        th1,th2 = model_params[0]
        F1 = [gm.Factor([X[i]],[0,th1[i]]).exp() for i in range(len(th1))]
        F2 = [gm.Factor([X[i],X[j]],[[0,0],[0,th2[i,j]]]).exp() for i,j in th2]
        model = gm.ising.Ising(F1+F2);
        model1 = copy.copy(model)
        model2 = copy.copy(model)

        plT = np.array([-3.23, -3.47, -2.98, -4.89, -2.28, -3.43, -6.22, -6.6 , -2.64,
            -3.27, -4.24, -2.28, -4.47, -3.45, -4.59, -2.37, -3.36, -4.47,
            -3.97, -2.37, -4.89, -3.23, -2.28, -2.37, -2.79, -2.65, -2.65,
            -5.22, -3.45, -2.79])

        res = gm.ising.refit_pll_opt(model1, D)
        self.assertTrue(close(res.fun, 3.563, 1e-2))
        pl = model1.pseudolikelihood(D); self.assertTrue( close(pl,plT, 2e-2) )

        res = gm.ising.refit_pll_sgd(model2, D, initStep=.1, maxIter=1000)
        pl = model2.pseudolikelihood(D); 
        self.assertEqual(len(pl) , len(plT))
        #assert( close(pl.mean(), -3.563, 1e-2) )   # average PLL should be about this
        #assert( close(pl,plT, 2e-2) )              # per-data PLL 
        # TODO: not checking due to convergence issues (?)


    #### Structure estimation  ##################################

    # Chow-Liu trees
    def testChowLiu(self):
        model = gm.ising.fit_chowliu( D )
        plT = np.array([-2.81, -4.97, -3.8 , -4.28, -1.63, -2.91, -4.64, -5.44, -2.66,
            -4.03, -3.99, -1.63, -4.71, -3.31, -4.46, -2.95, -2.41, -4.71,
            -3.9 , -2.95, -4.28, -2.81, -1.63, -2.95, -2.96, -2.96, -3.02,-5.35, -3.31, -2.96])
        pl = model.pseudolikelihood(D)
        self.assertTrue(close(pl, plT, 2e-2))

        plT = np.array([-3.36, -5.46, -4.15, -3.71, -2.02, -3.43, -4.59, -6.33, -3.45,
            -4.03, -4.4 , -2.02, -3.75, -4.31, -3.1 , -3.06, -2.99, -3.75,
            -3.68, -3.06, -3.71, -3.36, -2.02, -3.06, -2.88, -2.41, -2.49,-5.38, -4.31, -2.88])
        model3 = gm.ising.fit_chowliu( D , .03)
        pl = model3.pseudolikelihood(D)
        self.assertTrue(close(pl, plT, 2e-2))


    def testFitL1(self):
        model4 = gm.ising.fit_logregL1(D, 0.5)
        # TODO: check for "correctness"?  currently just crash on error

    def testFitMWeight(self):
        model5 = gm.ising.fit_mweight(D, 1e-2, .001, .1)
        # TODO: check for "correctness"?  currently just crash on error

    def testFitThresh(self):
        model6 = gm.ising.fit_threshold(D, .5, 4, 1e-2)
        # TODO: check for "correctness"?  currently just crash on error

    def testFitGreedy(self):
        model6 = gm.ising.fit_greedy(D, nnbr=2, threshold=0.05)  # use refit function?
        # TODO: check for "correctness"?  currently just crash on error




if __name__ == '__main__':
  unittest.main()


