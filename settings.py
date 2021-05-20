bgr_colors = {'r': (0, 0, 255),
              'g': (0, 255, 0),
              'b': (255, 0, 0),
              'c': (255, 255, 0),
              'm': (255, 0, 255),
              'y': (0, 255, 255),
              'w': (255, 255, 255),
              'k': (0, 0, 0)}

lines = [(0, 1, bgr_colors['c']),
         (0, 4, bgr_colors['c']),
         (1, 4, bgr_colors['c']),
         (0, 2, bgr_colors['y']),
         (1, 3, bgr_colors['y']),
         (4, 5, bgr_colors['b']),
         (5, 7, bgr_colors['b']),
         (5, 8, bgr_colors['m']),
         (8, 12, bgr_colors['r']),
         (12, 16, bgr_colors['g']),
         (5, 9, bgr_colors['m']),
         (9, 13, bgr_colors['r']),
         (13, 17, bgr_colors['g']),
         (7, 6, bgr_colors['b']),
         (6, 10, bgr_colors['m']),
         (10, 14, bgr_colors['r']),
         (14, 18, bgr_colors['g']),
         (6, 11, bgr_colors['m']),
         (11, 15, bgr_colors['r']),
         (15, 19, bgr_colors['g'])]

animal_classes = ['cat', 'cow', 'dog', 'horse', 'sheep']
activity_classes = ['stand', 'sit', 'lie', 'go', 'run']

target_size = (256, 256)
max_picture_shape = (1500, 800)
