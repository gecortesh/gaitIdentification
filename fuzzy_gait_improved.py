#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:01:45 2018

@author: gabych

Implementation from:
Chacon-Murguia, et al. (2013). 
A fuzzy scheme for gait cycle phase detection oriented to medical diagnosis.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def detect_phase(knee_sample, hip_sample, thigh_sample, time_sample, visualize=False): #stage_sample

    # New Antecedent/Consequent objects hold universe variables and membership
    # functions
    x_k = ctrl.Antecedent(np.arange(0, 1.1, .1), 'knee')
    x_h = ctrl.Antecedent(np.arange(0, 1.1, .1), 'hip')
    x_th = ctrl.Antecedent(np.arange(0, 1.1, .1), 'thigh')
    x_t = ctrl.Antecedent(np.arange(0, 1.1, .1), 'time')
#    x_s = ctrl.Antecedent(np.arange(0, 1.1, .1), 'stage')
    phase = ctrl.Consequent(np.arange(0, 9, 1), 'phase')
    
    # Custom membership functions 
    # movement intervals knee
    x_k['low'] = fuzz.gaussmf(x_k.universe, 0,0.175)
    x_k['medium']= fuzz.gaussmf(x_k.universe,0.5,0.175)
    x_k['high'] = fuzz.gaussmf(x_k.universe, 1,0.175)
    
    # movement intervals hip
    x_h['low'] = fuzz.gaussmf(x_h.universe, 0,0.175)
    x_h['medium'] = fuzz.gaussmf(x_h.universe, 0.5,0.175)
    x_h['high'] = fuzz.gaussmf(x_h.universe, 1,0.175)
    
    # toe off, heel strike detection using thigh and shank
#    x_th['TO-HS'] =  fuzz.trimf(x_th.universe, [0.40,0.599,0.99])
    x_th['TO-HS'] =  fuzz.smf(x_th.universe,0.05,0.489)
    
    # time intervals 
    x_t['low'] = fuzz.gaussmf(x_t.universe, 0,0.256)
    x_t['medium'] = fuzz.gaussmf(x_t.universe, 0.5,0.175)
    x_t['high'] = fuzz.gaussmf(x_t.universe, 1,0.175)
    
#    # stage intervals: loding respone LR, midstance MS, swing-stance SS
#    x_s['HS'] = fuzz.smf(x_s.universe,0.90,0.99)
#    x_s['S_LR'] = fuzz.zmf(x_s.universe, 0.1,0.11)
#    x_s['S_MS'] = fuzz.gaussmf(x_s.universe, 0.208,0.05)
#    x_s['S_SS'] = fuzz.smf(x_s.universe, 0.405,0.632)
#    
    # gait phases output
    phase['LR'] = fuzz.trimf(phase.universe,[0,1,2])
    phase['MSt'] = fuzz.trimf(phase.universe,[1,2,3])
    phase['TSt'] = fuzz.trimf(phase.universe,[2,3,4])
    phase['PSw'] = fuzz.trimf(phase.universe,[3,4,5])
    phase['ISw'] = fuzz.trimf(phase.universe,[4,5,6])
    phase['MSw'] = fuzz.trimf(phase.universe,[5,6,7])
    phase['TSw'] = fuzz.trimf(phase.universe,[6,7,8])
    
    # rules for phase detection
#    rule1 = ctrl.Rule(x_h['high'] & x_k['low'] & x_s['S_LR'] & x_t['low'], phase['LR'])
#    rule2 = ctrl.Rule(x_h['low'] | x_h['medium'] & x_k['low'] & x_s['S_MS'] & x_t['low'] & x_th['TO-HS'], phase['MSt'])
#    rule3 = ctrl.Rule(x_h['medium'] & x_k['low'] & x_s['S_MS'] & x_t['low'] & x_th['TO-HS'], phase['MSt'])
#    #rule4 = ctrl.Rule(x_h['low'] & x_k['low'] & x_t['medium'], phase['TSt'])
#    rule4 = ctrl.Rule(x_h['low'] & x_k['medium'] & x_t['medium'] & x_th['TO-HS'], phase['TSt'])
#    #rule5 = ctrl.Rule(x_h['low'] & x_k['high'] & x_s['S_SS'] & x_t['medium'], phase['PSw'])
#    rule5 = ctrl.Rule(x_h['low'] & x_k['high'] | x_k['medium'] & x_s['S_SS'] & x_t['medium'], phase['PSw'])
#    #rule6 = ctrl.Rule(x_h['medium'] & x_k['high'] & x_s['S_SS'] & x_t['medium'], phase['ISw'])
#    rule6 = ctrl.Rule(x_h['medium'] & x_k['high'] | x_k['medium'] & x_s['S_SS'] & x_t['medium'], phase['ISw'])
#    rule7 = ctrl.Rule(x_h['high'] & x_k['high']  & x_s['S_SS'] & x_t['high'], phase['MSw'])
#    rule8 = ctrl.Rule(x_h['high'] & x_k['low'] & x_s['S_SS'] & x_t['high'], phase['TSw'])
    
    rule1 = ctrl.Rule(x_h['high'] & x_k['low']  & x_t['low'], phase['LR'])
    rule2 = ctrl.Rule(x_h['low'] | x_h['medium'] & x_k['low'] &  x_t['low'], phase['MSt'])
    rule3 = ctrl.Rule(x_h['medium'] & x_k['low'] & x_t['low'] , phase['MSt'])
    #rule4 = ctrl.Rule(x_h['low'] & x_k['low'] & x_t['medium'], phase['TSt'])
    rule4 = ctrl.Rule(x_h['low'] & x_k['medium'] & x_t['medium'], phase['TSt'])
    #rule5 = ctrl.Rule(x_h['low'] & x_k['high'] & x_s['S_SS'] & x_t['medium'], phase['PSw'])
    rule5 = ctrl.Rule(x_h['low'] & x_k['high'] | x_k['medium'] &  x_t['medium'] & x_th['TO-HS'], phase['PSw'])
    #rule6 = ctrl.Rule(x_h['medium'] & x_k['high'] & x_s['S_SS'] & x_t['medium'], phase['ISw'])
    rule6 = ctrl.Rule(x_h['medium'] & x_k['high'] | x_k['medium'] & x_t['medium'] & x_th['TO-HS'], phase['ISw'])
    rule7 = ctrl.Rule(x_h['high'] & x_k['high']  & x_t['high'] & x_th['TO-HS'], phase['MSw'])
    rule8 = ctrl.Rule(x_h['high'] & x_k['low']  & x_t['high'] & x_th['TO-HS'], phase['TSw'])
    
    # control system
    phase_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])
    
    # View membership function definition and rules
#    x_k.view()
#    x_h.view()
#    x_t.view()
#    x_th.view()
##    x_s.view()
#    phase.view()
#    rule2.view()
    
    # simulation
    detecting_phase = ctrl.ControlSystemSimulation(phase_ctrl)
    
    # Pass inputs to the ControlSystem using Antecedent labels
    # Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
    
    detecting_phase.input['knee'] = float(knee_sample)
    detecting_phase.input['hip'] = float(hip_sample)
    detecting_phase.input['thigh'] = float(thigh_sample)
    detecting_phase.input['time'] = float(time_sample)
#    detecting_phase.input['stage'] = float(stage_sample)
    
    # Crunch the numbers
    detecting_phase.compute()
    
    # visualize
    if visualize == True:
        print (detecting_phase.output['phase'])
        phase.view(sim=detecting_phase)

    return detecting_phase.output['phase']
