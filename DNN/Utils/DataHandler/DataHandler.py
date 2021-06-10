import sys

import uproot
import pandas as pd
import numpy as np
from collections import defaultdict

import handler_kinematics as kinematics

from ROOT import TLorentzVector

def getParticlePt(p4):
    return p4.Pt()
def getParticleEta(p4):
    return p4.Eta()
def getParticlePhi(p4):
    return p4.Phi()

def getPt(p4):
    return [x.Pt() for x in p4]
def getEta(p4):
    return [x.Eta() for x in p4]
def getPhi(p4):
    return [x.Phi() for x in p4]

class DataHandler():
    ''' Class for manipulating LHE data. '''

    def __init__(self, lhe_file, tree_name, lorentz=False, rename={} ,variables=[]):

        print(">>> Initializing DataHandler ...")

        upfile = uproot.open(lhe_file)
        if len(variables) == 0: uptree = upfile[tree_name].arrays()
        else : uptree = upfile[tree_name].arrays(variables)
        self.pdarray = pd.DataFrame(uptree)
        map = dict((branch, branch.decode('utf-8')) for branch in self.pdarray.columns)
        self.pdarray = self.pdarray.rename(columns=map)
        # self.particles = defaultdict(list)
        self.pdarray.rename(columns=rename, inplace=True)
        for new_branch in rename.values(): 
            self.pdarray[new_branch] = pd.DataFrame(self.pdarray[new_branch].tolist(), index=self.pdarray.index)

        if lorentz:
            self.createParticle('w_bos')
            self.createParticle('mu')
            self.createParticle('v_mu')
            self.createParticle('el')
            self.createParticle('v_el')
            self.createParticle('q_fin')
            self.createParticle('q_init')
        else:
            self.branchSplitter('w_bos')
            self.branchSplitter('mu')
            self.branchSplitter('el')
            self.branchSplitter('v_el')
            self.branchSplitter('v_mu')
            self.branchSplitter('q_fin')
            self.branchSplitter('q_init')
            
    def branchSplitter(self, ptype):
        components = ['_px','_py','_pz','_E']
        for appendix in components:
            branch = ptype + appendix
            if branch not in self.pdarray.columns: continue
            multiplicity = len(self.pdarray[branch][0])
            if multiplicity == 1:
                variables = [branch]
            else:
                variables = []
                for i in range(multiplicity):
                    variables.append(branch + str(i+1))
            self.pdarray[variables] = pd.DataFrame(self.pdarray[branch].tolist(), index=self.pdarray.index)
            self.pdarray = self.pdarray.drop(labels=branch,axis=1)

    def createParticle(self, ptype):
        particles = defaultdict(list)
        components = ['_px', '_py', '_pz', '_E']
        for appendix in components:
            branch = ptype + appendix
            if branch not in self.pdarray.columns: continue
            multiplicity = len(self.pdarray[branch][0])
            if multiplicity == 1:
                variables = [branch]
                particles[ptype].append(branch)
            else:
                variables = []
                for i in range(multiplicity):
                    variables.append(branch+str(i+1))
                    particles[ptype+str(i+1)].append(branch+str(i+1))
            self.pdarray[variables] = pd.DataFrame(self.pdarray[branch].tolist(), index=self.pdarray.index)
        for particle, comp in particles.items():
            vector_list = []
            fourvectors = self.pdarray[comp]
            for p4 in fourvectors.values:
                vector_list.append(TLorentzVector(p4[0],p4[1],p4[2],p4[3]))
            self.pdarray[particle+'_p4'] = vector_list
        for appendix in components:
            branch = ptype + appendix
            if branch not in self.pdarray.columns: continue
            if multiplicity != 1: self.pdarray = self.pdarray.drop(labels=branch,axis=1)

    def defineParticle(self, components, particle):
        # if particle in self.particles: 
            # print('Error: ', particle, ' already defined')
            # sys.exit(1)
        # self.particles[particle] = components
        vector_list = []
        fourvectors = self.pdarray[components]
        for p4 in fourvectors.values:
            vector_list.append(TLorentzVector(p4[0], p4[1], p4[2], p4[3]))
        self.pdarray[particle + '_p4'] = vector_list

    def getPtEtaPhi(self, particle):
        # getPt = np.vectorize(getParticlePt)
        # getEta = np.vectorize(getParticleEta)
        # getPhi = np.vectorize(getParticlePhi)
        self.pdarray[particle+'_pt'] = getPt(self.pdarray[particle+'_p4'].values)
        self.pdarray[particle+'_eta'] = getEta(self.pdarray[particle+'_p4'].values)
        self.pdarray[particle+'_phi'] = getPhi(self.pdarray[particle+'_p4'].values)

    def appendBranch(self, branch, branch_name):
        self.pdarray[branch_name] = branch

    def getArray(self, variables):
        return self.pdarray[variables].values

    def filter(self, expression):
        self.pdarray = self.pdarray.query(expression)

    def appendQuadEqParams(self, flavour='mu'):
        a, b, c, delta = kinematics.abcdelta(self.pdarray, flavour)
        self.pdarray[flavour+'_a'] = pd.Series(a, index=self.pdarray.index)
        self.pdarray[flavour+'_b'] = pd.Series(b, index=self.pdarray.index)
        self.pdarray[flavour+'_c'] = pd.Series(c, index=self.pdarray.index)
        self.pdarray[flavour+'_delta'] = pd.Series(delta, index=self.pdarray.index)

    def appendTwoSolutions(self, flavour='mu'):
        sol0, sol1, label = kinematics.tag_solutions(self.pdarray, flavour)
        self.pdarray['v_'+flavour+'_sol0'] = pd.Series(sol0, index=self.pdarray.index)
        self.pdarray['v_'+flavour+'_sol1'] = pd.Series(sol1, index=self.pdarray.index)
        self.pdarray['v_'+flavour+'_label'] = pd.Series(label, index=self.pdarray.index)

    def appendEnergy(self, flavour):
        variables0 = ['v_'+flavour+'_px', 'v_'+flavour+'_py', 'v_'+flavour+'_sol0']
        variables1 = ['v_'+flavour+'_px', 'v_'+flavour+'_py', 'v_'+flavour+'_sol1']
        self.pdarray['v_'+flavour+'_predict_E0'] = kinematics.calc_energy(self.pdarray[variables0].values)
        self.pdarray['v_'+flavour+'_predict_E1'] = kinematics.calc_energy(self.pdarray[variables1].values)

    def calcCosTheta(self, variables, name):
        cosTheta = kinematics.cos_theta(self.pdarray, variables)
        self.pdarray[name+'_cos_theta'] = pd.Series(cosTheta, index=self.pdarray.index)

    def appendMass(self, variables, name):

        px_variables = []
        py_variables = []
        pz_variables = []
        e_variables = []
        
        for particle in variables:
            px_variables.append(particle[0])                
            py_variables.append(particle[1])
            pz_variables.append(particle[2])
            e_variables.append(particle[3])

        px = np.sum(self.pdarray[px_variables],1)
        py = np.sum(self.pdarray[py_variables], 1)
        pz = np.sum(self.pdarray[pz_variables], 1)
        e = np.sum(self.pdarray[e_variables], 1)

        self.pdarray[name+'_mass'] = kinematics.calc_mass(px,py,pz,e)

###################################################################################### new part

    def calcpL(self, variables, name):
        pL = kinematics.vector_manipulation(self.pdarray, variables)
        self.pdarray['pL'+name] = pd.Series(pL, index=self.pdarray.index)


    def appendSelectionCriteria(self, flavour='mu'):
        sel1, sel2, sel3, sel4, sel5, counter_1, counter_2, counter_3, counter_4, counter_5 = kinematics.tag_selectioncriteria(self.pdarray, flavour)
        self.pdarray['v_'+flavour+'_sel1'] = pd.Series(sel1, index=self.pdarray.index)
        self.pdarray['v_'+flavour+'_sel2'] = pd.Series(sel2, index=self.pdarray.index)
        self.pdarray['v_'+flavour+'_sel3'] = pd.Series(sel3, index=self.pdarray.index)
        self.pdarray['v_'+flavour+'_sel4'] = pd.Series(sel4, index=self.pdarray.index)
        self.pdarray['v_'+flavour+'_sel5'] = pd.Series(sel5, index=self.pdarray.index)
        self.pdarray['rnd_'+flavour+'_counter_1'] = pd.Series(counter_1, index=self.pdarray.index)
        self.pdarray['rnd_'+flavour+'_counter_2'] = pd.Series(counter_2, index=self.pdarray.index)
        self.pdarray['rnd_'+flavour+'_counter_3'] = pd.Series(counter_3, index=self.pdarray.index)
        self.pdarray['rnd_'+flavour+'_counter_4'] = pd.Series(counter_4, index=self.pdarray.index)
        self.pdarray['rnd_'+flavour+'_counter_5'] = pd.Series(counter_5, index=self.pdarray.index)


    def getPtvv(self,variables,name='pt_vv'):
        self.pdarray[name] = self.pdarray[variables[0]]+self.pdarray[variables[1]]
    def getPv_xx(self,variables,name='pv_xx'):
        self.pdarray[name] = self.pdarray[variables[0]]+self.pdarray[variables[1]]
    def getPv_yy(self,variables,name='pv_yy'):
        self.pdarray[name] = self.pdarray[variables[0]]+self.pdarray[variables[1]]

    def getMAOS(self, niter):
        mt2, p1x, p1y, p2x, p2y = kinematics.get_maos(self.pdarray, niter)   
        self.pdarray['mt2'] = mt2    
        self.pdarray['p1x'] = p1x    
        self.pdarray['p1y'] = p1y    
        self.pdarray['p2x'] = p2x    
        self.pdarray['p2y'] = p2y