#%%
import class_car as cp  # Importa a classe Car do módulo class_car
import numpy as np  # Biblioteca para manipulação de arrays
import cv2  # Biblioteca para processamento de imagens
import matplotlib.pyplot as plt  # Biblioteca para criação de gráficos
from sklearn.linear_model import RANSACRegressor  # Importa o RANSACRegressor para ajuste robusto de modelos

plt.rcParams['figure.figsize'] = (8, 6)  # Define o tamanho padrão das figuras do matplotlib

# Parâmetros globais
parameters = {
    'car_id': 0,  # ID do carro
    'ts': 5.0,  # Tempo de simulação
    'save': True,  # Se deve salvar os logs da simulação
    'logfile': 'logs/',  # Diretório para salvar os logs
}

# Modo de trajetória e controle
modo_trajetoria = 'circle'  # Define o modo de trajetória como circular
modo_controle = 'stanley'  # Define o modo de controle como Stanley

# Definir tipos de trajetórias
def trajectory_straight(car):
    # Função para trajetória reta
    num_points = 200  # Número de pontos da trajetória
    pos = car.getPos()  # Posição atual do carro
    x_ref = np.linspace(pos[0], pos[0] + 20, num_points)  # Coordenadas x da trajetória
    y_ref = np.ones_like(x_ref) * pos[1]  # Coordenadas y da trajetória (reta horizontal)
    nearest_index = np.argmin(np.abs(car.p[0] - x_ref))  # Encontra o ponto da trajetória mais próximo do carro
    return x_ref[nearest_index], y_ref[nearest_index]  # Retorna o ponto de referência mais próximo

def trajectory_circle(car):
    # Função para trajetória circular
    radius = 1  # Raio do círculo
    num_points = 200  # Número de pontos da trajetória
    pos = car.getPos()  # Posição atual do carro
    theta = np.linspace(0, 2 * np.pi, num_points)  # Ângulos para formar o círculo
    x_ref = pos[0] + radius * np.cos(theta)  # Coordenadas x do círculo
    y_ref = pos[1] + radius * np.sin(theta)  # Coordenadas y do círculo
    nearest_index = np.argmin(np.sqrt((car.p[0] - x_ref)**2 + (car.p[1] - y_ref)**2))  # Encontra o ponto da trajetória mais próximo do carro
    return x_ref[nearest_index], y_ref[nearest_index], theta[nearest_index]  # Retorna o ponto de referência mais próximo e o ângulo correspondente

# Seleciona a função de trajetória com base no modo escolhido
if modo_trajetoria == 'straight':
    trajectory_func = trajectory_straight
elif modo_trajetoria == 'circle':
    trajectory_func = trajectory_circle
else:
    raise ValueError("Modo de trajetória inválido. Escolha 'straight' ou 'circle'.")

def control_func(car):
    # Função de controle do carro
    global modo_trajetoria, modo_controle
    v_ref = 1.0  # Velocidade de referência

    # Obtém o ponto de referência da trajetória
    if modo_trajetoria == 'straight':
        x_ref, y_ref = trajectory_func(car)
    elif modo_trajetoria == 'circle':
        x_ref, y_ref, angle_ref = trajectory_func(car)

    if modo_controle == 'on-off':
        # Controle on-off
        if modo_trajetoria == 'straight':
            th_ref = 0.0  # Ângulo de referência para trajetória reta
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

    elif modo_controle == 'stanley':
        # Controle Stanley
        k_e = 1.0
        k_v = 0.5

        if modo_trajetoria == 'straight':
            th_ref = 0.0
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
            th_err = th_ref - car.getYaw()

        elif modo_trajetoria == 'circle':
            th_ref = angle_ref
            cross_track_error = (car.p[0] - x_ref) * np.cos(angle_ref) + (car.p[1] - y_ref) * np.sin(angle_ref)
            th_err = th_ref - car.getYaw()
            th_err = np.arctan2(np.sin(th_err), np.cos(th_err))

        delta = th_err + np.arctan2(k_e * cross_track_error, k_v * (v_ref - car.v))
        steer_cmd = np.clip(delta, -np.deg2rad(20.0), np.deg2rad(20.0))

    # Restringe os ângulos ao primeiro quadrante
    steer_cmd = np.clip(steer_cmd, -np.pi/2, np.pi/2)

    car.setVel(v_ref)  # Define a velocidade do carro
    car.setSteer(steer_cmd)  # Define o ângulo de esterçamento do carro

########################################
# Função de visão
########################################
def vision_func(car):
    # Função para processar a imagem da câmera do carro
    image = car.getImage()  # Obtém a imagem da câmera do carro
    image_draw = np.copy(image)  # Faz uma cópia da imagem para desenhar os resultados
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Converte a imagem para o espaço de cores HSV
    lower_yellow = np.array([28, 196, 242], dtype=np.uint8)  # Limite inferior para a cor amarela
    upper_yellow = np.array([68, 255, 255], dtype=np.uint8)  # Limite superior para a cor amarela
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)  # Cria uma máscara para a cor amarela
    yellow_mask = cv2.erode(yellow_mask, None, iterations=2)  # Erosão para remover ruídos
    yellow_mask = cv2.dilate(yellow_mask, None, iterations=2)  # Dilatação para restaurar a forma original
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Encontra os contornos na máscara

    if contours:
        contour = max(contours, key=cv2.contourArea)  # Seleciona o maior contorno
        model_ransac = RANSACRegressor(random_state=42)  # Modelo RANSAC para ajuste robusto
        X = contour[:, 0][:, 0].reshape(-1, 1)  # Coordenadas x do contorno
        y = contour[:, 0][:, 1]  # Coordenadas y do contorno
        model_ransac.fit(X, y)  # Ajusta o modelo RANSAC

        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])  # Ponto mais à esquerda do contorno
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])  # Ponto mais à direita do contorno
        middle_x = (leftmost[0] + rightmost[0]) // 2  # Coordenada x do ponto médio
        middle_y = (leftmost[1] + rightmost[1]) // 2  # Coordenada y do ponto médio

        cv2.drawContours(image_draw, [contour], -1, (0, 255, 0), 2)  # Desenha o contorno na imagem
        cv2.circle(image_draw, leftmost, 5, (255, 0, 0), -1)  # Desenha o ponto mais à esquerda
        cv2.circle(image_draw, rightmost, 5, (255, 0, 0), -1)  # Desenha o ponto mais à direita
        cv2.circle(image_draw, (middle_x, middle_y), 5, (0, 0, 255), -1)  # Desenha o ponto médio

        return image_draw, middle_x, middle_y  # Retorna a imagem processada e as coordenadas do ponto médio

    return image, car.getPos(), None  # Retorna a imagem original e a posição do carro se nenhum contorno for encontrado

########################################
# Executa a simulação
########################################
def run(parameters):
    plt.figure(1)
    plt.ion()  # Habilita modo interativo do matplotlib
    car = cp.Car(parameters)  # Cria uma instância do carro com os parâmetros fornecidos
    car.startMission()  # Inicia a missão do carro

    while car.t <= parameters['ts']:
        car.step()  # Atualiza o estado do carro
        control_func(car)  # Chama a função de controle
        image, _, _ = vision_func(car)  # Processa a imagem da câmera do carro
        plt.subplot(211)
        plt.cla()  # Limpa a figura
        plt.gca().imshow(image, origin='lower')  # Mostra a imagem processada
        plt.title('t = %.1f' % car.t)  # Exibe o tempo de simulação
        plt.subplot(212)
        plt.cla()  # Limpa a figura
        t = [traj['t'] for traj in car.traj]  # Coleta os tempos da trajetória do carro
        v = [traj['v'] for traj in car.traj]  # Coleta as velocidades da trajetória do carro
        plt.plot(t, v)  # Plota a velocidade em função do tempo
        plt.ylabel('v[m/s]')  # Legenda do eixo y
        plt.xlabel('t[s]')  # Legenda do eixo x
        plt.show()  # Mostra os gráficos
        plt.pause(0.01)  # Pausa para atualização dos gráficos

    car.stopMission()  # Finaliza a missão do carro

    if parameters['save']:
        car.save(parameters['logfile'])  # Salva os logs da simulação

    plt.ioff()  # Desabilita modo interativo do matplotlib
    print('Terminou...')

########################################
# Execução do código principal
########################################
if __name__ == "__main__":
    run(parameters)  # Executa a função principal com os parâmetros fornecidos

# %%

