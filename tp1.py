# %%
# -*- coding: utf-8 -*-
# Disciplina: Tópicos em Engenharia de Controle e Automação IV (ENG075): 
# Fundamentos de Veículos Autônomos - 2024/1
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automação
# DELT – Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################

import class_car as cp
import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.figure(1)
plt.ion()

########################################
# cria comunicação com o carrinho
car = cp.CarCoppelia()
v_ref = 1
fr_ant = 0
# começa a simulação
car.startMission()

u_vec = [0]
sum_err = 0
u = 0

# Posição e orientação iniciais
x0, y0 = car.p
th0 = car.th

# Modo de trajetória
modo_trajetoria = 'circle'  # Modo apenas para trajetória reta
modo_controle = 'stanley'  # Modo de controle: 'on-off' ou 'stanley'

# Definir tipos de trajetórias
def trajectory_straight(car):
    # Definir a trajetória em linha reta com deslocamento em y
    x_ref = x0 + 10  # Trajetória em linha reta com deslocamento em x
    y_ref = 4  # Posição fixa em y
    return x_ref, y_ref

def trajectory_circle(car):
    # Trajetória circular (não usada neste modo)
    radius = 2  # raio do círculo (ajustável)
    angular_speed = 0.1  # velocidade angular (ajustável)
    x_ref = x0 + radius * np.cos(angular_speed * car.t)
    y_ref = y0 + radius * np.sin(angular_speed * car.t)
    return x_ref, y_ref

# Escolha a trajetória desejada
if modo_trajetoria == 'straight':
    trajectory_func = trajectory_straight
elif modo_trajetoria == 'circle':
    trajectory_func = trajectory_circle
else:
    raise ValueError("Modo de trajetória inválido. Escolha 'straight' ou 'circle'.")

while car.t < 18:
    # lê senores
    car.step()

    # Definir posição de referência
    x_ref, y_ref = trajectory_func(car)

    # Calcular referência de orientação
    th_ref = np.arctan2(y_ref - car.p[1], x_ref - car.p[0])

    # Escolha do controlador
    if modo_controle == 'on-off':
        # Controlador lateral on-off
        th_err = th_ref - car.th
        if th_err > np.deg2rad(5):  # Se erro for maior que 5 graus
            steer_cmd = np.deg2rad(20.0)
        elif th_err < -np.deg2rad(5):  # Se erro for menor que -5 graus
            steer_cmd = -np.deg2rad(20.0)
        else:
            steer_cmd = 0  # Não estercar
    elif modo_controle == 'stanley':
        # Controlador Stanley
        k_e = 0.5  # Ganho de controle lateral
        k_v = 0.8  # Ganho de controle longitudinal

        # Erro de orientação
        th_err = th_ref - car.th

        # Erro lateral
        lateral_err = np.cross([np.cos(car.th), np.sin(car.th)], [x_ref - car.p[0], y_ref - car.p[1]])

        # Ângulo de referência para o controle Stanley
        delta = th_err + np.arctan2(k_e * lateral_err, k_v * (v_ref - car.v))

        # Limitar o ângulo de esterçamento
        steer_cmd = np.clip(delta, -np.deg2rad(30.0), np.deg2rad(30.0))

    else:
        raise ValueError("Modo de controle inválido. Escolha 'on-off' ou 'stanley'.")

    # Aplica o comando de esterçamento
    car.setSteer(steer_cmd)
    
    # Lei de controle
    err = (v_ref - car.v)
    if u < 0.5 or u > 0:
        sum_err += err
    u = 0.5*(err + 0.05*sum_err/2.5)

    car.setU(u)
    # pega imagem
    image = car.getImage()
    u_vec.append(u)

    t = [traj['t'] for traj in car.traj]
    v = [traj['v'] for traj in car.traj]

car.stopMission()
print('Terminou...')

# %%
plt.plot(t, u_vec)
plt.ylabel("u[m/s^2]")
plt.xlabel("t[s]")
plt.grid()
plt.savefig('u.pdf', format='pdf', bbox_inches='tight')
# %%
plt.plot(t, v)
plt.ylabel("v[m/s]")
plt.xlabel("t[s]")
plt.grid()
plt.savefig('v.pdf', format='pdf', bbox_inches='tight')
# %%

# %%