#%%
#%%
import class_car as cp
import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 6)

# Globais
parameters = {
    'car_id': 0,
    'ts': 5.0,  # tempo da simulacao
    'save': True,
    'logfile': 'logs/',
}

# Modo de trajetória e controle
modo_trajetoria = 'circle'  # Pode ser 'straight', 'circle' ou 'line_following'
modo_controle = 'on-off'  # Modo de controle: 'on-off' ou 'stanley'

# Definir tipos de trajetórias
def trajectory_straight(car):
    num_points = 200
    pos = car.getPos()
    x_ref = np.linspace(pos[0], pos[0] + 20, num_points)
    y_ref = np.ones_like(x_ref) * pos[1]
    nearest_index = np.argmin(np.abs(car.p[0] - x_ref))
    return x_ref[nearest_index], y_ref[nearest_index]

def trajectory_circle(car):
    # Parâmetros do círculo
    radius = 2.0  # Raio do círculo
    angular_speed = 1.0  # Velocidade angular (rad/s)
    v_ref = 1.0  # Velocidade de referência

    # Obter a posição e orientação atual do carro
    pos = car.getPos()
    t = car.t

    # Calcular a posição de referência no círculo
    angle_ref = angular_speed * t
    x_ref = radius * np.cos(angle_ref)
    y_ref = radius * np.sin(angle_ref)

    # Definir a velocidade desejada
    car.setVel(v_ref)

    return x_ref, y_ref, angle_ref

if modo_trajetoria == 'straight':
    trajectory_func = trajectory_straight
elif modo_trajetoria == 'circle':
    trajectory_func = trajectory_circle
elif modo_trajetoria == 'line_following':
    trajectory_func = None
else:
    raise ValueError("Modo de trajetória inválido. Escolha 'straight', 'circle' ou 'line_following'.")

########################################
# Função de controle
########################################
def control_func(car):
    global modo_trajetoria, modo_controle
    
    v_ref = 1.0  # Definir velocidade de referência como 1 m/s
    
    # Referências para controle
    if modo_trajetoria == 'line_following':
        image, middle_x, middle_y = vision_func(car)
        if middle_x is None or middle_y is None:
            return
        x_ref, y_ref = middle_x, middle_y
    else:
        if modo_trajetoria == 'straight':
            x_ref, y_ref = trajectory_func(car)
        elif modo_trajetoria == 'circle':
            x_ref, y_ref, angle_ref = trajectory_func(car)
    
    # Modo de controle on-off
    if modo_controle == 'on-off':
        th_ref = 0.0
        th_err = th_ref - car.getYaw()

        if modo_trajetoria == 'line_following':
            cross_track_error = (car.p[0] - x_ref) * np.cos(np.pi / 2 + car.getYaw()) + (car.p[1] - y_ref) * np.sin(np.pi / 2 + car.getYaw())
        else:
            if modo_trajetoria == 'straight':
                pos = car.getPos()
                x_start = pos[0]
                y_start = pos[1]
                x_end = pos[0] + 20
                y_end = pos[1]

                A = y_end - y_start
                B = -(x_end - x_start)
                C = x_end * y_start - y_end * x_start
                denom = np.sqrt(A ** 2 + B ** 2)
                cross_track_error = (A * car.p[0] + B * car.p[1] + C) / denom if denom != 0 else 0

            elif modo_trajetoria == 'circle':
                cross_track_error = (car.p[0] - x_ref) * np.cos(np.pi / 2 + car.getYaw()) + (car.p[1] - y_ref) * np.sin(np.pi / 2 + angle_ref)
        
        if cross_track_error > 0.1:
            steer_cmd = -np.deg2rad(20.0)
        elif cross_track_error < -0.1:
            steer_cmd = np.deg2rad(20.0)
        else:
            steer_cmd = 0

    # Modo de controle Stanley
    elif modo_controle == 'stanley':
        th_ref = 0.0

        k_e = 1.0
        k_v = 0.5

        th_err = th_ref - car.getYaw()

        if modo_trajetoria == 'line_following':
            cross_track_error = (car.p[0] - x_ref) * np.cos(np.pi / 2 + car.getYaw()) + (car.p[1] - y_ref) * np.sin(np.pi / 2 + car.getYaw())
        else:
            if modo_trajetoria == 'straight':
                pos = car.getPos()
                x_start = pos[0]
                y_start = pos[1]
                x_end = pos[0] + 20
                y_end = pos[1]

                A = y_end - y_start
                B = -(x_end - x_start)
                C = x_end * y_start - y_end * x_start
                denom = np.sqrt(A ** 2 + B ** 2)
                cross_track_error = abs((A * car.p[0] + B * car.p[1] + C) / denom) if denom != 0 else 0

            elif modo_trajetoria == 'circle':
                cross_track_error = (car.p[0] - x_ref) * np.cos(angle_ref) + (car.p[1] - y_ref) * np.sin(angle_ref)

        delta = th_err + np.arctan2(k_e * cross_track_error, k_v * (v_ref - car.v))

        if np.abs(th_err) > np.deg2rad(20):
            steer_cmd = np.deg2rad(20.0) * np.sign(th_err)
        else:
            steer_cmd = np.clip(delta, -np.deg2rad(20.0), np.deg2rad(20.0))

    car.setVel(v_ref)  # Define a velocidade
    car.setSteer(steer_cmd)

########################################
# Função de visão para seguir a linha amarela
########################################
def vision_func(car):
    image = car.getImage()
    image_draw = np.copy(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    
    # Criar máscara para detectar amarelo
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    
    # Erosão e dilatação para limpar a máscara
    yellow_mask = cv2.erode(yellow_mask, None, iterations=2)
    yellow_mask = cv2.dilate(yellow_mask, None, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Encontrar o contorno com a maior área (presumindo que seja a faixa amarela)
        contour = max(contours, key=cv2.contourArea)
        
        # Encontrar os pontos mais à esquerda e à direita do contorno
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        
        # Calcular o ponto do meio da faixa amarela
        middle_x = (leftmost[0] + rightmost[0]) // 2
        middle_y = (leftmost[1] + rightmost[1]) // 2

        # Desenhar contornos e pontos na imagem para visualização
        cv2.drawContours(image_draw, [contour], -1, (0, 255, 0), 2)
        cv2.circle(image_draw, leftmost, 5, (255, 0, 0), -1)
        cv2.circle(image_draw, rightmost, 5, (255, 0, 0), -1)
        cv2.circle(image_draw, (middle_x, middle_y), 5, (0, 0, 255), -1)

        return image_draw, middle_x, middle_y

    return image, None, None

########################################
# Executa a simulação
########################################
def run(parameters):
    plt.figure(1)
    plt.ion()
    
    car = cp.Car(parameters)
    car.startMission()

    while car.t <= parameters['ts']:
        car.step()
        if modo_trajetoria == 'line_following':
            image, _, _ = vision_func(car)
        else:
            image = car.getImage()
        control_func(car)
        
        plt.subplot(211)
        plt.cla()
        plt.gca().imshow(image, origin='lower')
        plt.title('t = %.1f' % car.t)
        
        plt.subplot(212)
        plt.cla()
        t = [traj['t'] for traj in car.traj]
        v = [traj['v'] for traj in car.traj]
        plt.plot(t, v)
        plt.ylabel('v[m/s]')
        plt.xlabel('t[s]')
        
        plt.show()
        plt.pause(0.01)

    car.stopMission()

    if parameters['save']:
        car.save(parameters['logfile'])
    
    plt.ioff()
    print('Terminou...')

########################################
# Execução do código principal
########################################
if __name__=="__main__":
    run(parameters)

# %%
