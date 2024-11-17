import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np

# Средние и стандартные отклонения для нормализации
mean = np.array([0.6796, 0.6407, 0.6611])
std = np.array([0.2572, 0.2562, 0.2543])


# Трансформация для тестирования и валидации
def get_transform1():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

# Преобразование изображения для предсказания


def preprocess_image1(image, transform):
    """
    Загрузка изображения, применение трансформаций и возвращение тензора.

    Args:
        image_path (str): Путь к изображению.
        transform (torchvision.transforms.Compose): Трансформация для изображения.

    Returns:
        torch.Tensor: Изображение, преобразованное в тензор.
    """
    try:
        # Если image - это строка (путь), загрузите его как изображение
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        # Если это объект PIL.Image.Image, просто продолжите
        elif not isinstance(image, Image.Image):
            raise ValueError("Ожидался путь или объект PIL.Image.Image")
        image_tensor = transform(image).unsqueeze(
            0)  # Добавление batch dimension
        return image_tensor
    except Exception as e:
        raise ValueError(f"Ошибка при загрузке изображения: {e}")
