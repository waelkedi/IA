#!/usr/bin/env python
# -*- coding: utf-8 -*-


import rospy
import math
import geometry
import clustering
from data_buffer import DataBuffer

# topic /sensors/odom
from nav_msgs.msg import Odometry

# topic /darknet_ros/bounding_boxes
from darknet_ros_msgs.msg import BoundingBoxes

# topic /scan
from sensor_msgs.msg import LaserScan
# scan.range[179] = distance between the robot and an object


object_searched = ["stop sign",
                   "backpack" ,
                   "refrigerator",
                   "motorbike",
                   "pottedplant",
                   "suitcase",
                   "teddy bear"]

odom_bag = DataBuffer(maxlen=1000)
bounding_boxes_bag = DataBuffer()
scan_bag = DataBuffer(maxlen=1000)

objects = []
previous_robot_position = None
previous_object = None

def _process_bounding_box(bounding_box):
    global previous_robot_position, previous_object
    current_robot_position = odom_bag.get_closest_to(bounding_box.image_header.stamp)
    distance_from_object = scan_bag.get_closest_to(bounding_box.image_header.stamp)
    # TODO: checker si c'est le même object ou un autre
    if previous_robot_position is not None and previous_object is not None:
        object_position = geometry.compute_object_coordinates(previous_robot_position.x,
                                            previous_robot_position.y,
                                            current_robot_position.x,
                                            current_robot_position.y,
                                            distance_from_object)
        classes = [i.Class for i in bounding_box.bounding_boxes]
        objects.append((object_position, classes))
    previous_robot_position = current_robot_position
    previous_object = bounding_box


def _process_pending_bounding_boxes():
    global bounding_boxes_bag
    for i in range(len(bounding_boxes_bag)): # Try to process each objet only once.
        current_bounding_box = bounding_boxes_bag.pop()
        try:
            _process_bounding_box(current_bounding_box)
        except Exception as e:
            bounding_boxes_bag.appendleft(current_bounding_box)
            continue

def odom_handler(odom):
    odom_bag.append(odom.header.stamp,  odom.pose.pose.position)
    _process_pending_bounding_boxes()

def bounding_box_handler(bounding_box):
    try:
        _process_bounding_box(bounding_box)
    except BufferError as e:
        bounding_boxes_bag.append(bounding_box.image_header.stamp, bounding_box)


def scan_handler(scan):
    distance = scan.ranges[179]
    if scan.range_min < distance < scan.range_max:
        scan_bag.append(scan.header.stamp, distance)


def write_results(res):
    file_ = open("results.txt", 'w')
    file_.write(str(res))
    file_.close()


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('object_finder', anonymous=True)

    rospy.Subscriber("/sensors/odom", Odometry, odom_handler)
    rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, bounding_box_handler)
    rospy.Subscriber("/scan", LaserScan, scan_handler)
    # spin() simply keeps python from exiting until this node is stopped

    rospy.spin()
    organized_data = clustering.organize_data(objects, object_searched)
    object_positions = clustering.find_object_position(organized_data)
    write_results(object_positions)


if __name__ == '__main__':
    listener()
