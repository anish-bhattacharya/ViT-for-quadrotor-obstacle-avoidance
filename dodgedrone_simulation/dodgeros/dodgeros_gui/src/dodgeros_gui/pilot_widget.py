#!/usr/bin/env python
import os

import rospy
from python_qt_binding import loadUi
from scipy import interpolate

try:
    # Starting from Qt 5 QWidget is defined in QtWidgets and not QtGui anymore
    from python_qt_binding.QtWidgets import QWidget, QGraphicsOpacityEffect, QFileDialog
    from python_qt_binding.QtGui import QFont
except:
    from python_qt_binding.QtGui import QWidget, QFont, QGraphicsOpacityEffect
from python_qt_binding.QtCore import QTimer, Slot

import geometry_msgs.msg as geometry_msgs
import dodgeros_msgs.msg as dodgeros_msgs
from nav_msgs.msg import Path, Odometry
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
from std_srvs.srv import Trigger, SetBool

import math
import numpy as np
import traceback


class PilotWidget(QWidget):

    def __init__(self, parent):
        # Init QWidget
        super(PilotWidget, self).__init__(parent)
        self.setObjectName('Autopilot Widget')

        # set variables
        self._quad_namespace = None
        self._connected = False
        self._traj_filename = None

        # publishers for the logic
        self._arm_bridge_pub = None
        self._start_pub = None
        self._land_pub = None
        self._off_pub = None
        self._force_hover_pub = None

        self._go_to_pose_pub = None
        self._sampled_trajectory_pub = None

        self._reference_sub = None
        self._odometry_sub = None
        self._active_reference = Path()
        self._active_reference_stamp = rospy.Time.now()
        self._odometry = Odometry()
        self._odometry_stamp = rospy.Time.now()
        self._telemetry = dodgeros_msgs.Telemetry()
        self.click_response_text = ""
        self.click_response_success = True
        self.reference_msg = None
        # now = rospy.Time.now()
        # five_secs_ago = now - rospy.Duration(5)
        self.click_response_dur = 2.0
        # bef = rospy.Duration(self.click_response_dur)
        self.click_response_time = rospy.Time.now()

        # load UI
        ui_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../../resource/pilot_widget.ui')
        loadUi(ui_file, self)

        # Timer
        self._update_info_timer = QTimer(self)
        self._update_info_timer.timeout.connect(self.update_gui)
        self._update_info_timer.start(100)

        self.disconnect()

    def connect(self, quad_namespace):
        self._quad_namespace = quad_namespace

        self._arm_bridge_pub = rospy.Publisher(
            quad_namespace + '/dodgeros_pilot/enable', std_msgs.Bool, queue_size=1)
        self._start_pub = rospy.Publisher(
            quad_namespace + '/dodgeros_pilot/start', std_msgs.Empty, queue_size=1)
        self._land_pub = rospy.Publisher(
            quad_namespace + '/dodgeros_pilot/land', std_msgs.Empty, queue_size=1)
        self._off_pub = rospy.Publisher(
            quad_namespace + '/dodgeros_pilot/off', std_msgs.Empty, queue_size=1)
        self._force_hover_pub = rospy.Publisher(
            quad_namespace + '/dodgeros_pilot/force_hover', std_msgs.Empty,
            queue_size=1)

        self._go_to_pose_pub = rospy.Publisher(
            quad_namespace + '/dodgeros_pilot/go_to_pose', geometry_msgs.PoseStamped,
            queue_size=1)
        self._sampled_trajectory_pub = rospy.Publisher(
            quad_namespace + '/dodgeros_pilot/trajectory', dodgeros_msgs.Reference,
            queue_size=1)
        self._reference_sub = rospy.Subscriber(
            quad_namespace + '/active/path',
            Path, self.active_reference_cb)
        self._odometry_sub = rospy.Subscriber(
            quad_namespace + '/dodgeros_pilot/odometry',
            Odometry, self.odometry_cb)
        self._telemetry_sub = rospy.Subscriber(
            quad_namespace + '/dodgeros_pilot/telemetry',
            dodgeros_msgs.Telemetry, self.telemetry_cb)

        self.button_arm_bridge.setEnabled(True)
        self.button_start.setEnabled(False)
        self.button_land.setEnabled(True)
        self.button_off.setEnabled(True)
        self.button_force_hover.setEnabled(True)
        self.button_go_to_pose.setEnabled(True)

        self._connected = True

    def disconnect_pub_sub(self, pub):
        if pub is not None:
            pub.unregister()
            pub = None

    def disconnect_service(self, serv):
        if serv is not None:
            serv.close()
            serv = None

    def disconnect(self):
        self.disconnect_pub_sub(self._arm_bridge_pub)
        self.disconnect_pub_sub(self._start_pub)
        self.disconnect_pub_sub(self._land_pub)
        self.disconnect_pub_sub(self._off_pub)
        self.disconnect_pub_sub(self._force_hover_pub)
        self.disconnect_pub_sub(self._go_to_pose_pub)

        self.button_arm_bridge.setEnabled(False)
        self.button_start.setEnabled(False)
        self.button_land.setEnabled(False)
        self.button_off.setEnabled(False)
        self.button_force_hover.setEnabled(False)
        self.button_go_to_pose.setEnabled(False)

        self._connected = False

    def active_reference_cb(self, msg):
        self._active_reference = msg
        self._active_reference_stamp = rospy.Time.now()

    def odometry_cb(self, msg):
        self._odometry = msg
        self._odometry_stamp = rospy.Time.now()

    def telemetry_cb(self, msg):
        self._telemetry = msg

    def update_gui(self):
        if self._connected:
            self.bridge_name.setText('%s' % self._telemetry.bridge_type.data)
            self.control_computation_time.setText('%.1f' % (0.0))  # TODO
            self.trajectory_execution_left_duration.setText('%.1f' % max(0.0, self._telemetry.reference_left_duration))
            self.trajectories_left_in_queue.setText('%d' % self._telemetry.num_references_in_queue)  # TODO

            if self._telemetry.bridge_armed.data:
                self.bridge_state.setStyleSheet('QLabel { color : orange; }')
                self.bridge_state.setText('Armed')
                self.button_start.setEnabled(True)
            else:
                self.bridge_state.setStyleSheet('QLabel { color : black; }')
                self.bridge_state.setText('Disarmed')
                self.button_start.setEnabled(False)

            self.pilot_title.setText('Pilot')
            self.background_widget.setStyleSheet('QWidget { background-color:none }')

            # Low-Level status
            if self._telemetry.voltage > 15.0:
                self.status_battery_voltage.setStyleSheet('QLabel { color : green; }')
            elif self._telemetry.voltage > 14.0:
                self.status_battery_voltage.setStyleSheet('QLabel { color : orange; }')
            else:
                self.status_battery_voltage.setStyleSheet('QLabel { color : red; }')

            self.status_battery_voltage.setText('%.2f' % self._telemetry.voltage)

            # State Estimate
            self.state_est_frame_id.setText(
                'Frame ID: %s' % self._odometry.header.frame_id)
            euler_angles = np.squeeze(np.rad2deg(self.quat_to_euler_angles(self._odometry.pose.pose.orientation)))
            ref_euler_angles = np.squeeze(
                np.rad2deg(self.quat_to_euler_angles(self._telemetry.reference.pose.orientation)))
            self.state_est_position.setText(
                'x:{:6.2f} y:{:6.2f} z:{:6.2f}'.format(self._odometry.pose.pose.position.x,
                                                       self._odometry.pose.pose.position.y,
                                                       self._odometry.pose.pose.position.z))
            self.state_est_velocity.setText(
                'x:{:6.2f} y:{:6.2f} z:{:6.2f}'.format(self._odometry.twist.twist.linear.x,
                                                       self._odometry.twist.twist.linear.y,
                                                       self._odometry.twist.twist.linear.z))
            self.state_est_orientation.setText(
                'r:{:6.2f} p:{:6.2f} h:{:6.2f}'.format(euler_angles[0], euler_angles[1], euler_angles[2]))
            self.state_est_body_rates.setText(
                'x:{:6.2f} y:{:6.2f} z:{:6.2f}'.format(self._odometry.twist.twist.angular.x / math.pi * 180.0,
                                                       self._odometry.twist.twist.angular.y / math.pi * 180.0,
                                                       self._odometry.twist.twist.angular.z / math.pi * 180.0))

            # Reference State
            if len(self._active_reference.poses) > 0:
                self.ref_position.setText(
                    'x:{:6.2f} y:{:6.2f} z:{:6.2f}'.format(self._telemetry.reference.pose.position.x,
                                                           self._telemetry.reference.pose.position.y,
                                                           self._telemetry.reference.pose.position.z))
                self.ref_velocity.setText(
                    'x:{:6.2f} y:{:6.2f} z:{:6.2f}'.format(self._telemetry.reference.velocity.linear.x,
                                                           self._telemetry.reference.velocity.linear.y,
                                                           self._telemetry.reference.velocity.linear.z))
                self.ref_orientation.setText(
                    'r:{:6.2f} p:{:6.2f} h:{:6.2f}'.format(ref_euler_angles[0], ref_euler_angles[1],
                                                           ref_euler_angles[2]))
                self.ref_body_rates.setText(
                    'x:{:6.2f} y:{:6.2f} z:{:6.2f}'.format(
                        self._telemetry.reference.velocity.angular.x / math.pi * 180.0,
                        self._telemetry.reference.velocity.angular.y / math.pi * 180.0,
                        self._telemetry.reference.velocity.angular.z / math.pi * 180.0))
            else:
                self.ref_position.setText('x: none  y:none  z:none')
                self.ref_velocity.setText('x: none  y:none  z:none')
                self.ref_orientation.setText('r: none  p:none  h:none')
                self.ref_body_rates.setText('x: none  y:none  z:none')

            # Go-to-pose Panel
            if (self._telemetry.reference_left_duration > 0.0 and not math.isinf(
                    self._telemetry.reference_left_duration)) or self._telemetry.num_references_in_queue == 0:
                self.button_go_to_pose.setEnabled(False)
                self.go_to_pose_x.setEnabled(False)
                self.go_to_pose_y.setEnabled(False)
                self.go_to_pose_z.setEnabled(False)
                self.go_to_pose_heading.setEnabled(False)
                self.go_to_pose_x.setText('%.2f' % self._telemetry.reference.pose.position.x)
                self.go_to_pose_y.setText('%.2f' % self._telemetry.reference.pose.position.y)
                self.go_to_pose_z.setText('%.2f' % self._telemetry.reference.pose.position.z)
                self.go_to_pose_heading.setText('%.2f' % ref_euler_angles[2])
            else:
                self.button_go_to_pose.setEnabled(True)
                self.go_to_pose_x.setEnabled(True)
                self.go_to_pose_y.setEnabled(True)
                self.go_to_pose_z.setEnabled(True)
                self.go_to_pose_heading.setEnabled(True)

            # Load trajectory Panel
            if math.isinf(self._telemetry.reference_left_duration) and self.reference_msg is not None:
                if (rospy.get_rostime().to_sec() - self.reference_msg.points[0].state.t > 2.0):
                    self.button_start_trajectory.setEnabled(True)

            # Click Status
            time_diff = rospy.Time.now() - self.click_response_time
            if time_diff.to_sec() > 2.0:
                self.click_status.setText("")
            else:
                self.click_status.setText(self.click_response_text)
                self.click_status.setFont(QFont('Monospace', 10, QFont.Monospace))
                opacity_effect = QGraphicsOpacityEffect()
                opacity_effect.setOpacity(1.0 - time_diff.to_sec() / 2.0)
                # disappearing text
                self.click_status.setGraphicsEffect(opacity_effect)
                if self.click_response_success:
                    self.click_status.setStyleSheet('QLabel { color : green; }')
                else:
                    self.click_status.setStyleSheet('QLabel { color : red; }')

        else:
            # Autopilot status
            self.control_computation_time.setText('0.0')
            self.trajectory_execution_left_duration.setText('0.0')
            self.trajectories_left_in_queue.setText('0')

            # Low-Level status
            self.status_battery_voltage.setStyleSheet('QLabel { color : gray; }')
            self.status_battery_voltage.setText('Not Available')

            # State Estimate
            self.state_est_frame_id.setText('Frame ID:')
            self.state_est_position.setText('Not Available')
            self.state_est_position.setFont(QFont('Monospace', 10, QFont.Monospace))
            self.state_est_velocity.setText('Not Available')
            self.state_est_velocity.setFont(QFont('Monospace', 10, QFont.Monospace))
            self.state_est_orientation.setText('Not Available')
            self.state_est_orientation.setFont(QFont('Monospace', 10, QFont.Monospace))
            self.state_est_body_rates.setText('Not Available')
            self.state_est_body_rates.setFont(QFont('Monospace', 10, QFont.Monospace))

            # Reference State
            self.ref_position.setText('Not Available')
            self.ref_position.setFont(QFont('Monospace', 10, QFont.Monospace))
            self.ref_velocity.setText('Not Available')
            self.ref_velocity.setFont(QFont('Monospace', 10, QFont.Monospace))
            self.ref_orientation.setText('Not Available')
            self.ref_orientation.setFont(QFont('Monospace', 10, QFont.Monospace))
            self.ref_body_rates.setText('Not Available')
            self.ref_body_rates.setFont(QFont('Monospace', 10, QFont.Monospace))

            # Click Status
            self.click_status.setText("")

    @Slot(bool)
    def on_button_connect_clicked(self):
        if (self._connected):
            self.disconnect()
            self.button_connect.setText('Connect')
        else:
            quad_namespace = self.namespace_text.text()
            self.connect(quad_namespace)
            self.button_connect.setText('Disconnect')

    @Slot(bool)
    def on_button_arm_bridge_clicked(self):
        arm_message = std_msgs.Bool(True)
        self._arm_bridge_pub.publish(arm_message)

    @Slot(bool)
    def on_button_start_clicked(self):
        start_message = std_msgs.Empty()
        self._start_pub.publish(start_message)

    @Slot(bool)
    def on_button_land_clicked(self):
        land_message = std_msgs.Empty()
        self._land_pub.publish(land_message)

    @Slot(bool)
    def on_button_off_clicked(self):
        off_message = std_msgs.Empty()
        self._off_pub.publish(off_message)
        arm_message = std_msgs.Bool(False)
        self._arm_bridge_pub.publish(arm_message)

    @Slot(bool)
    def on_button_force_hover_clicked(self):
        force_hover_msg = std_msgs.Empty()
        self._force_hover_pub.publish(force_hover_msg)

    @Slot(bool)
    def on_button_go_to_pose_clicked(self):
        try:
            go_to_pose_msg = geometry_msgs.PoseStamped()
            go_to_pose_msg.pose.position.x = float(self.go_to_pose_x.text())
            go_to_pose_msg.pose.position.y = float(self.go_to_pose_y.text())
            go_to_pose_msg.pose.position.z = float(self.go_to_pose_z.text())

            heading = float(self.go_to_pose_heading.text()) / 180.0 * math.pi

            go_to_pose_msg.pose.orientation.w = math.cos(heading / 2.0)
            go_to_pose_msg.pose.orientation.z = math.sin(heading / 2.0)

            self._go_to_pose_pub.publish(go_to_pose_msg)

        except:
            rospy.logwarn("Could not read and send go to pose message!")

    @staticmethod
    def quat_to_euler_angles(q):
        #  Computes the euler angles from a unit quaternion using the
        #  z-y-x convention.
        #  euler_angles = [roll pitch yaw]'
        #  A quaternion is defined as q = [qw qx qy qz]'
        #  where qw is the real part.
        euler_angles = np.zeros((3, 1))
        euler_angles[0] = np.arctan2(
            2 * q.w * q.x + 2 * q.y * q.z, q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z)
        euler_angles[1] = -np.arcsin(2 * q.x * q.z - 2 * q.w * q.y)
        euler_angles[2] = np.arctan2(
            2 * q.w * q.z + 2 * q.x * q.y, q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z)
        return euler_angles

    @staticmethod
    def resample_by_interpolation(time, signal, new_dt):
        if (new_dt == -1):
            return signal.T, time

        if (signal.size == 0):
            return signal.T, time

        time = time.reshape(1, -1).squeeze()

        new_signal = []

        # signal can have different columns with data gathered with the same time vector
        for signal_i in signal.T:
            signal_i = signal_i.reshape(1, -1).squeeze()

            f_interpolate = interpolate.interp1d(time, signal_i)

            new_time = np.arange(time[0], time[-1], new_dt)
            new_signal_i = f_interpolate(new_time)
            new_signal.append(new_signal_i)

        return np.array(new_signal), new_time
