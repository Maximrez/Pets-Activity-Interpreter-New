from PIL import ImageOps


def max_confidence(probabilities):
    m_prob = 0.
    m_idx = 0
    for idx in range(len(probabilities)):
        if probabilities[idx] > m_prob:
            m_prob = probabilities[idx]
            m_idx = idx
    return m_idx, m_prob


def image_padding(image, target_padding, target_size):
    w_size, h_size = image.size

    x_padding = int(target_padding[0] * w_size / (target_size[0] - 2 * target_padding[0]))
    y_padding = int(target_padding[1] * h_size / (target_size[1] - 2 * target_padding[1]))

    image = ImageOps.expand(image, border=(x_padding, y_padding), fill='white')
    return image, x_padding, y_padding


def rescale(x, y, crop_x1, crop_x2, crop_y1, crop_y2, x_padding, y_padding, target_size):
    x = int(x * (crop_x2 - crop_x1 + 2 * x_padding) / target_size[0]) - x_padding + crop_x1
    y = int(y * (crop_y2 - crop_y1 + 2 * y_padding) / target_size[1]) - y_padding + crop_y1
    return x, y
