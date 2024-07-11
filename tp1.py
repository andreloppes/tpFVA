#%%
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
import matplotlib.pyplot as plt

plt.figure(1)
plt.ion()

########################################
# Cria comunicação com o carrinho
car = cp.CarCoppelia()
v_ref = 1.0  # Velocidade de referência inicial

# Começa a simulação
car.startMission()

# Lista para armazenar o erro de orientação ao longo do tempo
th_err_list = []

# Modo de trajetória
modo_trajetoria = 'straight'  # Pode ser 'straight' ou 'circle'
modo_controle = 'stanley'  # Modo de controle: 'on-off' ou 'stanley'

# Posição e orientação iniciais
x0, y0 = car.p
th0 = car.th

# Referência de orientação inicial para a trajetória reta
th_ref = th0  # Inicialmente a mesma orientação que o carro

# Definir tipos de trajetórias
def trajectory_straight(car):
    # Parâmetros da reta
    x_start = x0
    y_start = y0
    x_end = x0 + 20  # Comprimento da reta em x
    
    # Coeficientes da equação da reta ax + by + c = 0
    A = y_end - y_start
    B = -(x_end - x_start)
    C = x_end * y_start - y_end * x_start
    
    # Gerar um linspace ao longo da reta y = y0 com comprimento 20 em x
    num_points = 200
    x_ref = np.linspace(x_start, x_end, num_points)
    y_ref = (-A * x_ref - C) / B  # Calcular y para cada x usando a equação da reta
    
    # Encontrar o ponto mais próximo na trajetória
    nearest_index = np.argmin(np.abs(car.p[0] - x_ref))
    
    return x_ref[nearest_index], y_ref[nearest_index]

def trajectory_circle(car):
    # Trajetória circular contínua
    radius = 1  # Raio do círculo (ajustável)
    angular_speed = 0.5  # Velocidade angular (ajustável)
    
    # Posição inicial como centro do círculo
    x_center, y_center = x0, y0
    
    # Calcular o ângulo desejado para o controle Stanley
    angle_ref = th0 + angular_speed * car.t
    
    # Calcular a posição de referência usando a equação paramétrica do círculo
    x_ref = x_center + radius * np.cos(angle_ref)
    y_ref = y_center + radius * np.sin(angle_ref)
    
    return x_ref, y_ref, angle_ref


# Escolha a trajetória desejada
if modo_trajetoria == 'straight':
    trajectory_func = trajectory_straight
elif modo_trajetoria == 'circle':
    trajectory_func = trajectory_circle
else:
    raise ValueError("Modo de trajetória inválido. Escolha 'straight' ou 'circle'.")

while car.t < 36:
    # Lê sensores e atualiza estado do carro
    car.step()

    # Definir posição de referência
    if modo_trajetoria == 'circle':
        x_ref, y_ref, angle_ref = trajectory_func(car)
    else:
        x_ref, y_ref = trajectory_func(car)

    # Calcular referência de orientação
    th_ref = np.arctan2(y_ref - car.p[1], x_ref - car.p[0])

    # Restringir th_ref ao intervalo de -pi a pi
    th_ref = np.mod(th_ref, 2 * np.pi)
    if th_ref > np.pi:
        th_ref -= 2 * np.pi

    # Escolha do controlador
    if modo_controle == 'on-off':
        # Controlador lateral on-off com histerese mínima
        th_err = th_ref - car.th

        # Calcular crosstrack error como a distância perpendicular à trajetória
        if modo_trajetoria == 'straight':
            # Para trajetória reta, calcular a distância perpendicular entre o carro (car.p) e a trajetória (x_ref, y_ref)
            x_start = x0
            y_start = y0
            x_end = x0 + 20
            y_end = y0

            # Fórmula da distância ponto a linha
            A = y_end - y_start
            B = -(x_end - x_start)
            C = x_end * y_start - y_end * x_start
            denom = np.sqrt(A**2 + B**2)
            if denom != 0:
                cross_track_error = (A * car.p[0] + B * car.p[1] + C) / denom
            else:
                cross_track_error = 0

        elif modo_trajetoria == 'circle':
            # Para trajetória circular, crosstrack error é ajustado conforme a trajetória circular
            cross_track_error = (car.p[0] - x_ref) * np.cos(np.pi/2 + th0) + (car.p[1] - y_ref) * np.sin(np.pi/2 + th0)

        if cross_track_error > 0.1:  # Histerese mínima para evitar oscilações próximas ao zero
            steer_cmd = -np.deg2rad(20.0)
        elif cross_track_error < -0.1:
            steer_cmd = np.deg2rad(20.0)
        else:
            steer_cmd = 0  # Não esterçar

    elif modo_controle == 'stanley':
        # Controlador Stanley
        k_e = 1.0  # Ganho de controle lateral 
        k_v = 0.5  # Ganho de controle longitudinal 

        # Erro de orientação
        th_err = th_ref - car.th     

        # Calcular crosstrack error baseado na trajetória específica
        if modo_trajetoria == 'straight':
            # Para trajetória reta, calcular a distância perpendicular entre o carro (car.p) e a trajetória (x_ref, y_ref)
            x_start = x0
            y_start = y0
            x_end = x0 + 20
            y_end = y0

            # Fórmula da distância ponto a linha
            A = y_end - y_start
            B = -(x_end - x_start)
            C = x_end * y_start - y_end * x_start
            denom = np.sqrt(A**2 + B**2)
            if denom != 0:
                cross_track_error = abs(A * car.p[0] + B * car.p[1] + C) / denom
            else:
                cross_track_error = 0
        elif modo_trajetoria == 'circle':
            cross_track_error = (car.p[0] - x_ref) * np.cos(angle_ref) + (car.p[1] - y_ref) * np.sin(angle_ref)

        # Ângulo de referência para o controle Stanley
        delta = th_err + np.arctan2(k_e * cross_track_error, k_v * (v_ref - car.v))

        # Limitar o ângulo de esterçamento baseado no erro de orientação e no crosstrack error
        if np.abs(th_err) > np.deg2rad(20):
            steer_cmd = np.deg2rad(20.0) * np.sign(th_err)
        else:
            steer_cmd = np.clip(delta, -np.deg2rad(20.0), np.deg2rad(20.0))

    else:
        raise ValueError("Modo de controle inválido. Escolha 'on-off' ou 'stanley'.")

    # Aplicar o comando de esterçamento
    car.setSteer(steer_cmd)
    
    # Setar velocidade constante
    car.setVel(v_ref)

    # Obter imagem
    image = car.getImage()

    # Armazenar o erro de orientação para análise posterior
    th_err_list.append(th_err)

car.stopMission()
print('Terminou...')

# %%
