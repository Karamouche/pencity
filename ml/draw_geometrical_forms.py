import cv2
import numpy as np
import random as rd

from misc import convert_to_absolute





def draw_shapes_on_image(background, list_of_tensors, BACKGROUND_SIZE):
    shape_functions = [draw_random_line, draw_random_polygon, draw_bezier_curve]  # Add more shape functions here if needed
    num_shapes = rd.randint(0, 7)  # Decide how many shapes to try adding

    # Try adding new shapes
    for _ in range(num_shapes):
        shape_func = rd.choice(shape_functions)
        # Update background with the result from try_adding_shape
        background = try_adding_shape(background, shape_func, list_of_tensors, BACKGROUND_SIZE)
        
    background = add_noise_to_margins(background, margin=100, random_noise_level = True, noise_level_min=0.001, noise_level_max=0.01)


    return background




def try_adding_shape(original_background, shape_func, existing_bboxes, BACKGROUND_SIZE, max_attempts=10):
    for _ in range(max_attempts):
        background_copy = original_background.copy()
        new_background, shape_bbox = shape_func(background_copy, BACKGROUND_SIZE)
        if not check_shape_overlap(shape_bbox, existing_bboxes, BACKGROUND_SIZE):
            return new_background
    return  original_background # Return the original background if no suitable position was found


def draw_random_line(background, BACKGROUND_SIZE, exclude_center=True, margin=100):
    height, width = background.shape[:2]
    # Zones are defined based on the central exclusion square
    zones = {
        'left': ((0, margin), (0, height)),
        'right': ((width - margin, width), (0, height)),
        'top': ((margin, width - margin), (0, margin)),
        'bottom': ((margin, width - margin), (height - margin, height))
    }
    
    # Randomly select a zone to draw the line in
    zone_selected = rd.choice(list(zones.keys()))
    x_range, y_range = zones[zone_selected]

    # Generate random start and end points within the selected zone
    start_point = (rd.randint(*x_range), rd.randint(*y_range))
    end_point = (rd.randint(*x_range), rd.randint(*y_range))

    # Varying thickness
    thickness = rd.randint(1, 30)

    # Draw the line on the background
    cv2.line(background, start_point, end_point, 255, thickness)

    # Calculate bounding box in absolute coordinates
    bbox = calculate_bounding_box([start_point, end_point])

    return background, bbox
    



def draw_random_polygon(background, BACKGROUND_SIZE, exclude_center=True, margin=100):
    height, width = background.shape[:2]
    # Zones are defined based on the central exclusion square
    zones = {
        'left': ((0, margin), (0, height)),
        'right': ((width - margin, width), (0, height)),
        'top': ((margin, width - margin), (0, margin)),
        'bottom': ((margin, width - margin), (height - margin, height))
    }

    # Randomly select a zone to draw the polygon in
    zone_selected = rd.choice(list(zones.keys()))
    x_range, y_range = zones[zone_selected]

    num_sides = np.random.randint(3, 7)  # 3 to 6 sides
    points = []

    for _ in range(num_sides):
        # Generate random points within the selected zone
        x = rd.randint(*x_range)
        y = rd.randint(*y_range)
        points.append((x, y))

    # Calculate bounding box in absolute coordinates
    bbox = calculate_bounding_box(points)
    
    # Draw the polygon on the background
    cv2.fillPoly(background, [np.array(points)], 255)

    return background, bbox




def draw_bezier_curve(background, BACKGROUND_SIZE, exclude_center=True, margin=100):
    height, width = background.shape[:2]
    # Zones are defined based on the central exclusion square
    zones = {
        'left': ((0, margin), (0, height)),
        'right': ((width - margin, width), (0, height)),
        'top': ((margin, width - margin), (0, margin)),
        'bottom': ((margin, width - margin), (height - margin, height))
    }

    # Randomly select a zone to draw the Bezier curve in
    zone_selected = rd.choice(list(zones.keys()))
    x_range, y_range = zones[zone_selected]

    # Define control points within the selected zone
    control_points = [(rd.randint(*x_range), rd.randint(*y_range)) for _ in range(3)]

    # Calculate points of the Bezier curve
    curve_points = []
    for t in np.linspace(0, 1, num=100):
        x = (1-t)**2 * control_points[0][0] + 2*(1-t)*t*control_points[1][0] + t**2 * control_points[2][0]
        y = (1-t)**2 * control_points[0][1] + 2*(1-t)*t*control_points[1][1] + t**2 * control_points[2][1]
        curve_points.append((int(x), int(y)))

    # Calculate bounding box in absolute coordinates
    bbox = calculate_bounding_box(curve_points)

    # Draw the Bezier curve on the background
    for point in curve_points:
        cv2.circle(background, point, 1, 255, -1)


    return background, bbox

        

def add_noise_to_margins(background, margin=100, random_noise_level=True, noise_level_min=0.001, noise_level_max=0.01):
    """
    Add salt and pepper noise to the margins of the image.

    Parameters:
    - background: The image to which to add noise.
    - margin: Width of the margin where noise is to be added.
    - noise_level: The proportion of pixels in the margin areas that will be altered.
    """
    #get the noise level if random
    if random_noise_level:
        noise_level = rd.uniform(noise_level_min, noise_level_max)
    else:
        noise_level = noise_level_max
        
    height, width = background.shape[:2]
    
    # Define margin areas (top, bottom, left, right)
    margin_areas = [
        (0, margin, 0, width),  # Top margin
        (height - margin, height, 0, width),  # Bottom margin
        (margin, height - margin, 0, margin),  # Left margin
        (margin, height - margin, width - margin, width)  # Right margin
    ]
    
    for area in margin_areas:
        for i in range(area[0], area[1]):
            for j in range(area[2], area[3]):
                if rd.random() < noise_level:
                    # Randomly choose to make the pixel white or black
                    background[i, j] = 255 if rd.random() < 0.5 else 0

    return background



def calculate_bounding_box(points):
    """
    Calculate the bounding box for a given set of points.

    Parameters:
    - points: A list of tuples, where each tuple represents the coordinates of a point (x, y).

    Returns:
    - A list containing the bounding box in absolute coordinates: [center_x, center_y, width, height].
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)

    bbox_center_x = (x_min + x_max) // 2  # Use integer division for absolute coordinates
    bbox_center_y = (y_min + y_max) // 2
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    return [bbox_center_x, bbox_center_y, bbox_width, bbox_height]


def check_shape_overlap(shape_bbox, existing_bboxes, background_size):
    """
    Check if the proposed shape's bounding box overlaps with any existing objects,
    assuming all bounding boxes are in absolute coordinates.
    """
    shape_top_left = (shape_bbox[0] - shape_bbox[2] // 2, shape_bbox[1] - shape_bbox[3] // 2)
    shape_bottom_right = (shape_bbox[0] + shape_bbox[2] // 2, shape_bbox[1] + shape_bbox[3] // 2)
    

    for bbox in existing_bboxes:
        # Assuming existing_bboxes are already in absolute coordinates
        x, y, h, w = convert_to_absolute(bbox, background_size)
        bbox_top_left = (x - h // 2, y - w // 2)
        bbox_bottom_right = (x + h // 2, y + w // 2)

        # Check for overlap
        if (shape_top_left[0] < bbox_bottom_right[0] and shape_bottom_right[0] > bbox_top_left[0] and
            shape_top_left[1] < bbox_bottom_right[1] and shape_bottom_right[1] > bbox_top_left[1]):
            return True  # Overlap detected
    return False


