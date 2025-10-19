import numpy as np
from lvgp_bayes.utils.variables import NumericalVariable,CategoricalVariable
from lvgp_bayes.utils.input_space import InputSpace

def borehole():
    V = np.array(np.meshgrid(
        np.linspace(0.05,0.15,4),
        np.linspace(700,820,4)
    )).T.reshape(-1,2)

    var_names = ['r_w','H_l']
    maps = {
        j:{
        var_names[k]:V[j,k] for k in range(2)
        } for j in range(V.shape[0])
    }

    config = InputSpace()
    r = NumericalVariable(name='r',lower=100,upper=50000)
    Tu = NumericalVariable(name='T_u',lower=63070,upper=115600)
    Hu = NumericalVariable(name='H_u',lower=990,upper=1110)
    Tl = NumericalVariable(name='T_l',lower=63.1,upper=116)
    L = NumericalVariable(name='L',lower=1120,upper=1680)
    K_w = NumericalVariable(name='K_w',lower=9855,upper=12045)
    config.add_inputs([r,Tu,Hu,Tl,L,K_w])

    config.add_input(
        CategoricalVariable(name='t',levels=np.arange(V.shape[0]))
    )
    return config,V,maps

def otl():
    V = np.array(np.meshgrid(
        np.linspace(0.5,3,6),
        np.linspace(50,300,3)
    )).T.reshape(-1,2)

    var_names = ['Rf','B']
    maps = {
        j:{
        var_names[k]:V[j,k] for k in range(2)
        } for j in range(V.shape[0])
    }

    config = InputSpace()
    Rb1 = NumericalVariable(name='Rb1',lower=50,upper=150)
    Rb2 = NumericalVariable(name='Rb2',lower=25,upper=70)
    Rc1 = NumericalVariable(name='Rc1',lower=1.2,upper=2.5)
    Rc2 = NumericalVariable(name='Rc2',lower=0.25,upper=1.20)
    config.add_inputs([Rb1,Rb2,Rc1,Rc2])
    config.add_input(
        CategoricalVariable(name='t',levels=np.arange(V.shape[0]))
    )
    return config,V,maps

def piston():
    V = np.array(np.meshgrid(
        np.linspace(1000,5000,5),
        np.linspace(90000,110000,4)
    )).T.reshape(-1,2)

    var_names = ['k','P_0']
    maps = {
        j:{
        var_names[k]:V[j,k] for k in range(2)
        } for j in range(V.shape[0])
    }

    config = InputSpace()
    M = NumericalVariable(name='M',lower=30,upper=60)
    S = NumericalVariable(name='S',lower=0.005,upper=0.02)
    V0 = NumericalVariable(name='V_0',lower=0.002,upper=0.01)
    Ta = NumericalVariable(name='T_a',lower=290,upper=296)
    T0 = NumericalVariable(name='T_0',lower=340,upper=360)
    config.add_inputs([M,S,V0,Ta,T0])
    config.add_input(
        CategoricalVariable(name='t',levels=np.arange(V.shape[0]))
    )
    return config,V,maps