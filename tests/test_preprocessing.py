import cv2
import numpy as np

from src.preprocessing import preprocess_image


def test_preprocess_image(tmp_path):
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    # Create a simple colored object
    cv2.circle(image, (100, 100), 60, (0, 180, 255), -1)

    image_path = tmp_path / "test.jpg"
    cv2.imwrite(str(image_path), image)

    processed = preprocess_image(str(image_path))

    assert processed is not None
    assert processed.shape == image.shape
    assert processed.dtype == np.uint8
