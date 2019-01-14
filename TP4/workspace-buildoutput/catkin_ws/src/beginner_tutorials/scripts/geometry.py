import math

"""
The authors of this code is the group of Nico Salamone and Bastien Vanderplaetse.
"""

def compute_angle(x1, y1, x2, y2):
    """
    Compute the angle between two points. The returned value is expressed in radians and is between -pi and pi.

    :x1: The abscissa of the first point.
    :y1: The ordinate of the first point.
    :x2: The abscissa of the second point.
    :y2: The ordinate of the second point.
    :return: The angle between the two points in radians (between -pi and pi).
    """

    # Change the origin of the Euclidean space.
    y = y2 - y1
    x = x2 - x1

    # Compute the angle.
    angle = math.atan2(y, x)

    return angle

def compute_object_coordinates(prev_x, prev_y, curr_x, curr_y, dist_robot_obj):
    """
    Compute the coordinates of a object from the previous and the current position of the robot and the distance between
    the robot and the object.

    :prev_x: The previous abscissa of the robot.
    :prev_y: The previous ordinate of the robot.
    :curr_x: The current abscissa of the robot.
    :prev_y: The current ordinate of the robot.
    :dist_robot_obj: The distance between the robot and the object.
    :return: The coordinates of the object.
    """

    # Compute the direction of the robot.
    angle = compute_angle(prev_x, prev_y, curr_x, curr_y)

    # Compute the abscissa and the ordinate of the object.
    delta_x = dist_robot_obj * math.cos(angle)
    delta_y = dist_robot_obj * math.sin(angle)

    obj_x = curr_x + (delta_x)
    obj_y = curr_y + (delta_y)

    return obj_x, obj_y

#####

if __name__ == '__main__':
    x1, y1 = (0, 0)
    x2, y2 = (-3, 1)

    angle = math.degrees(compute_angle(x1, y1, x2, y2))
    print(angle)

    #####

    prev_x, prev_y = 4, -2
    curr_x, curr_y = 3, 4
    dist_robot_obj = 6
    obj_x, obj_y = compute_object_coordinates(prev_x, prev_y, curr_x, curr_y, dist_robot_obj)
    print("x: {}\ny: {}".format(obj_x, obj_y))
