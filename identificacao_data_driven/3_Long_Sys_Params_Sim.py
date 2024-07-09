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

import numpy as np
import pandas as pd
import math
import control as ctrl
import matplotlib.pyplot as plt

########################################
coastdown_data = pd.read_csv(r"./Coast_Down_Test.csv")
coastdown_data_clipped = coastdown_data.loc[(coastdown_data["t"] >= 5.2) & 
                                            (coastdown_data["t"] <= 18.3), ["t", "v"]]

time     = coastdown_data_clipped["t"].values
velocity = coastdown_data_clipped["v"].values

v0 = velocity[0]
time -= time[0]
Tf = time[-1]
time /= Tf
velocity /= v0

# Check in plot
fig = plt.figure(1)
plt.plot(time, velocity)
plt.show()

# %%
########################################
beta = 0.4
mass = 6.35      # [kg]
rho = 1.225     # [kg/m^3]
Af = 0.06       # [m^2]

CdAf = (2*mass*beta*math.atan(beta)) / (v0*Tf*rho)
Cd = CdAf/Af

print(CdAf)
print(Cd)

# %%
########################################
Rx = (v0*mass*math.atan(beta)) / (beta*Tf)
print(Rx)

# %%
########################################
nu = (1/beta)*(np.tan((1-time)*math.atan(beta)))

# Check in plot
fig = plt.figure(2)
plt.plot(time, velocity)
plt.plot(time, nu)
plt.show()

# %%
########################################
# Check beta value
beta_check = v0*math.sqrt((rho*CdAf) / (2*Rx))
print(beta_check)

# %%
##############################################
##############################################

step_resp_data = pd.read_csv(r"./Syst_Ident_Test.csv")
step_resp_data_clipped = step_resp_data.loc[(step_resp_data["t"] >= 1.75) & 
                                            (step_resp_data["t"] <= 3.1), 
                                            ["t", "v", "u"]]

time     = step_resp_data_clipped["t"].values
u        = step_resp_data_clipped["u"].values
velocity = step_resp_data_clipped["v"].values

# Check in plot
# fig = plt.figure(3)
# plt.plot(time, velocity)
# plt.plot(time, u)
# plt.show()

# fig, ax1 = plt.subplots(figsize=(8, 8))
# ax2 = ax1.twinx()

# ax1.plot(time, u, color = "#69b3a2", lw = 3)
# ax2.plot(time, velocity, color = "#3399e6", lw = 4)

# ax1.set_xlabel("t (s)")
# ax1.set_ylabel("u (m/s^2)", color = "#69b3a2", fontsize = 14)
# ax1.tick_params(axis = "y", labelcolor = "#69b3a2")

# ax2.set_ylabel("velocity (m/s)", color = "#3399e6", fontsize = 14)
# ax2.tick_params(axis = "y", labelcolor = "#3399e6")


###################################################################


# Plotting figure 1
plt.figure(1)
plt.plot(time, u, label='Sinal de entrada: Degrau')
plt.plot(time, velocity, label='Resposta ao degrau')
plt.xlabel('Tempo t (em s)')
plt.ylabel('Accelerac. u(t) (em m/s2)', color='blue')
plt.legend()
plt.grid(True)

# Plotting figure 2
plt.figure(2)
plt.plot(time, u, label='Sinal de entrada: Degrau')
plt.plot(time, velocity, label='Resposta ao degrau')
dy = np.diff(velocity) / np.diff(time)
k = 50-27
tang = (time - time[k]) * dy[k] + velocity[k]
plt.plot(time, tang, '--', label='Tangente')
plt.axhline(0, color='red', linestyle='--')
plt.axvline(2, color='red', linestyle='--')
plt.xlabel('Tempo t (em s)')
plt.ylabel('Accelerac. u(t) (em m/s2)', color='blue')
plt.legend()
plt.grid(True)
plt.savefig('Ident_time_cte.pdf', format = 'pdf', bbox_inches = 'tight')

# Plotting identified model
K_c = 1.6
tau_c = 0.03 # 0.002
# td_c = 0.04
Gc_withDelay = ctrl.TransferFunction([K_c], [tau_c, 1, 0])
Gc_noDelay = ctrl.TransferFunction([K_c], [tau_c, 1, 0])

# print(Gc_withDelay)
print(Gc_noDelay)

# Simulate response of the identified model
u_delayed = u.copy() # generates a shallow copied list/array
u_delayed[26:29] = 0
_, velocity_estim = ctrl.forced_response(Gc_withDelay, time, u_delayed)

# Plotting figure 3
plt.figure(3)
plt.plot(time, u, label='uo(t) (m/s^2)')
plt.plot(time, velocity, label='Resp. Original')
plt.plot(time, velocity_estim, label='Resp. Estimada')
plt.xlabel('t (s)')
plt.ylabel('v(t) (m/s)', color = 'blue')
plt.legend()
plt.grid(True)
plt.savefig('Ident_validating.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()


# %%
