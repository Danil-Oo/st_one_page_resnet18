import streamlit as st
import torch
import time
from PIL import Image
import matplotlib.pyplot as plt
from models.cell_resnet18 import CellsResNet, load_model1
from models.cell_preprocessing import get_transform1, preprocess_image1
from models.sport_resnet18 import MyFreezeResNet, load_model
from models.sport_preprocessing import get_transform, preprocess_image
import requests
from io import BytesIO


# Сначала инициализируем переменные и напишем функции для корректной работы модели и дальнейших предсказаний и выводов
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPORT_MODEL_PATH = 'models/sport_resnet18.pth'
CELLS_MODEL_PATH = 'models/cells_resnet18.pth'

sport_class_names = {
    0: 'air hockey',
    1: 'ampute football',
    2: 'archery',
    3: 'arm wrestling',
    4: 'axe throwing',
    5: 'balance beam',
    6: 'barell racing',
    7: 'baseball',
    8: 'basketball',
    9: 'baton twirling',
    10: 'bike polo',
    11: 'billiards',
    12: 'bmx',
    13: 'bobsled',
    14: 'bowling',
    15: 'boxing',
    16: 'bull riding',
    17: 'bungee jumping',
    18: 'canoe slamon',
    19: 'cheerleading',
    20: 'chuckwagon racing',
    21: 'cricket',
    22: 'croquet',
    23: 'curling',
    24: 'disc golf',
    25: 'fencing',
    26: 'field hockey',
    27: 'figure skating men',
    28: 'figure skating pairs',
    29: 'figure skating women',
    30: 'fly fishing',
    31: 'football',
    32: 'formula 1 racing',
    33: 'frisbee',
    34: 'gaga',
    35: 'giant slalom',
    36: 'golf',
    37: 'hammer throw',
    38: 'hang gliding',
    39: 'harness racing',
    40: 'high jump',
    41: 'hockey',
    42: 'horse jumping',
    43: 'horse racing',
    44: 'horseshoe pitching',
    45: 'hurdles',
    46: 'hydroplane racing',
    47: 'ice climbing',
    48: 'ice yachting',
    49: 'jai alai',
    50: 'javelin',
    51: 'jousting',
    52: 'judo',
    53: 'lacrosse',
    54: 'log rolling',
    55: 'luge',
    56: 'motorcycle racing',
    57: 'mushing',
    58: 'nascar racing',
    59: 'olympic wrestling',
    60: 'parallel bar',
    61: 'pole climbing',
    62: 'pole dancing',
    63: 'pole vault',
    64: 'polo',
    65: 'pommel horse',
    66: 'rings',
    67: 'rock climbing',
    68: 'roller derby',
    69: 'rollerblade racing',
    70: 'rowing',
    71: 'rugby',
    72: 'sailboat racing',
    73: 'shot put',
    74: 'shuffleboard',
    75: 'sidecar racing',
    76: 'ski jumping',
    77: 'sky surfing',
    78: 'skydiving',
    79: 'snow boarding',
    80: 'snowmobile racing',
    81: 'speed skating',
    82: 'steer wrestling',
    83: 'sumo wrestling',
    84: 'surfing',
    85: 'swimming',
    86: 'table tennis',
    87: 'tennis',
    88: 'track bicycle',
    89: 'trapeze',
    90: 'tug of war',
    91: 'ultimate',
    92: 'uneven bars',
    93: 'volleyball',
    94: 'water cycling',
    95: 'water polo',
    96: 'weightlifting',
    97: 'wheelchair basketball',
    98: 'wheelchair racing',
    99: 'wingsuit flying'
}

cells_class_names = {0: 'EOSINOPHIL',
                     1: 'LYMPHOCYTE', 2: 'MONOCYTE', 3: 'NEUTROPHIL'}


sport_model = load_model(SPORT_MODEL_PATH, DEVICE)
cells_model = load_model1(CELLS_MODEL_PATH, DEVICE)


# Функция предиктов
def predict_function(model, image, device):
    if model == sport_model:
        image = preprocess_image(image, get_transform()).to(device)

        model.eval()
        with torch.inference_mode():
            start_time = time.time()
            pred = sport_class_names[model(image).argmax().item()]
            proba = round(model(image).softmax(dim=1).max().item(), 4)
            elapsed_time = round(time.time() - start_time, 4)

        return pred, proba, elapsed_time, image
    else:
        image = preprocess_image1(image, get_transform1()).to(device)

        model.eval()
        with torch.inference_mode():
            start_time = time.time()
            pred = cells_class_names[model(image).argmax().item()]
            proba = round(model(image).softmax(dim=1).max().item(), 4)
            elapsed_time = round(time.time() - start_time, 4)

        return pred, proba, elapsed_time, image

 # Функция загрузки изображения по URL


def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Не удалось загрузить изображение: {e}")
        return None


page = st.sidebar.radio('Страницы:', ('Спорт', 'Клетки крови'))

if page == 'Спорт':
    st.title('Определение вида спорта по изображению')
    st.divider()
    st.subheader('Модель: ResNet18')
    st.divider()
    st.subheader('График функции потерь и метрики точности по эпохам:')
    st.image('images/learning_sport.png')
    st.subheader('Тепловая карта предсказаний:')
    st.image('images/heatmap_sport.png')
    st.write('F1 score: 0.8524')
    st.divider()

    load_option = st.radio(
        'Выберите способ загрузки изображения', ('По ссылке', 'Файлом'))

    if load_option == 'По ссылке':
        url = st.text_input("Введите ссылку")
        if url:
            image = load_image_from_url(url)
            if image:
                st.image(image)
                pred, proba, el_time, _ = predict_function(
                    sport_model, image, DEVICE)
                st.write(f'Предсказанный класс: {pred}')
                st.write(f'Вероятность: {proba}')
                st.write(f'Скорость предсказания: {el_time}')
    else:
        uploaded_files = st.file_uploader(
            'Вставьте изображение', accept_multiple_files=True)
        if uploaded_files:
            images = [Image.open(uploaded_file)
                      for uploaded_file in uploaded_files]
            for i, img in enumerate(images):
                st.image(
                    img, caption=f'Изображение {i + 1}', use_column_width=True)
                pred, proba, el_time, _ = predict_function(
                    sport_model, img, DEVICE)
                st.write(f'Предсказанный класс: {pred}')
                st.write(f'Вероятность: {proba}')
                st.write(f'Скорость предсказания: {el_time}')

if page == 'Клетки крови':
    st.title('Определение клетки крови по изображению')
    st.divider()
    st.subheader('Модель: ResNet18')
    st.divider()
    st.subheader('График функции потерь и метрики точности по эпохам:')
    st.image('images/learning_cells.png')
    st.subheader('Тепловая карта предсказаний:')
    st.image('images/heat_map_cells.jpg')
    st.write('F1 score: 0.8211')
    st.divider()

    load_option = st.radio(
        'Выберите способ загрузки изображения', ('По ссылке', 'Файлом'))

    if load_option == 'По ссылке':
        url = st.text_input("Введите ссылку")
        if url:
            image = load_image_from_url(url)
            if image:
                st.image(image)
                pred, proba, el_time, _ = predict_function(
                    cells_model, image, DEVICE)
                st.write(f'Предсказанный класс: {pred}')
                st.write(f'Вероятность: {proba}')
                st.write(f'Скорость предсказания: {el_time}')
    else:
        uploaded_files = st.file_uploader(
            'Вставьте изображение', accept_multiple_files=True)
        if uploaded_files:
            images = [Image.open(uploaded_file)
                      for uploaded_file in uploaded_files]
            for i, img in enumerate(images):
                st.image(
                    img, caption=f'Изображение {i + 1}', use_column_width=True)
                pred, proba, el_time, _ = predict_function(
                    cells_model, img, DEVICE)
                st.write(f'Предсказанный класс: {pred}')
                st.write(f'Вероятность: {proba}')
                st.write(f'Скорость предсказания: {el_time}')
