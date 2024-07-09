# %%
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 21:54:49 2023
@author: Emerson A.S.

# Disciplina: Tópicos em Engenharia de Controle e Automação IV (ENG075): 
# Fundamentos de Veículos Autônomos - 2023/2
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automação
# DELT – Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
"""

import class_car as cp
import matplotlib.pyplot as plt
import pandas as pd

########################################
plt.ion()
plt.figure(1)

########################################
# communication with the simulator
car = cp.CarCoppelia()
car.startMission()

########################################
# simulation loop
while car.t < 22:
	
    # read sensors
    car.step()
	
    # lateral control: steering angle
    # delta = -2.0*(0.0 - car.p[0]) + 5.0*(0.0 - car.w)
    # car.setSteer(-delta)
    car.setSteer(0)
    
    # actuation
    if car.t < 5.0:
        car.setVel(2)
    else:
        car.setU(0.0)
	
    # plota
    # plt.clf()
    t = [traj['t'] for traj in car.traj]
    v = [traj['v'] for traj in car.traj]
    # plt.plot(t,v)
    # plt.show()
    # plt.pause(0.01)


fig = plt.figure(2)
plt.plot(t,v)
plt.xlabel('t')
plt.ylabel('v')
plt.savefig('Coast_Down_Test.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()

# Save simulation data
# car.traj is a Python dictionary of lists of float values
coast_down_test_data = pd.DataFrame(car.traj)
coast_down_test_data.to_csv(r".\Coast_Down_Test.csv", index = False, header = True)

print('Terminou...')


# %%
