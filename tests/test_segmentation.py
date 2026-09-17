import cv2
import numpy as np

from src.preprocessing import preprocess_image
from src.segmentation import segment_foreground, detect_blemishes


def test_segmentation(tmp_path):
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    # Fruit-like object
    cv2.circle(image, (100, 100), 60, (0, 180, 255), -1)

    # Dark spot / blemish
    cv2.circle(image, (120, 100), 10, (20, 20, 20), -1)

    image_path = tmp_path / "fruit.jpg"
    cv2.imwrite(str(image_path), image)

    processed = preprocess_image(str(image_path))

    fruit_mask = segment_foreground(processed)

    assert fruit_mask is not None
    assert fruit_mask.shape[:2] == processed.shape[:2]

    blemish_mask, defect_ratio, contours = detect_blemishes(
        processed,
        fruit_mask
    )

    assert blemish_mask is not None
    assert 0 <= defect_ratio <= 1
    assert isinstance(contours, list)
