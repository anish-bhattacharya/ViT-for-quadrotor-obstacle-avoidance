#!/usr/bin/python3

import numpy as np


class AgileCommandMode(object):
    """Defines the command type."""

    # Control individual rotor thrusts.
    SRT = 0
    # Specify collective mass-normalized thrust and bodyrates.
    CTBR = 1
    # Command linear velocities. The linear velocity is expressed in world frame.
    LINVEL = 2

    def __new__(cls, value):
        """Add ability to create CommandMode constants from a value."""
        if value == cls.SRT:
            return cls.SRT
        if value == cls.CTBR:
            return cls.CTBR
        if value == cls.LINVEL:
            return cls.LINVEL

        raise ValueError("No known conversion for `%r` into a command mode" % value)


class AgileCommand:
    def __init__(self, mode):
        self.mode = AgileCommandMode(mode)
        self.t = 0.0

        # SRT functionality
        self.rotor_thrusts = [0.0, 0.0, 0.0, 0.0]

        # CTBR functionality
        self.collective_thrust = 0.0
        self.bodyrates = [0.0, 0.0, 0.0]

        # LINVEL functionality
        self.velocity = [0.0, 0.0, 0.0]
        self.yawrate = 0.0


class AgileQuadState:
    def __init__(self, quad_state):
        self.t = quad_state.t

        self.pos = np.array(
            [
                quad_state.pose.position.x,
                quad_state.pose.position.y,
                quad_state.pose.position.z,
            ],
            dtype=np.float32,
        )
        self.att = np.array(
            [
                quad_state.pose.orientation.w,
                quad_state.pose.orientation.x,
                quad_state.pose.orientation.y,
                quad_state.pose.orientation.z,
            ],
            dtype=np.float32,
        )
        self.vel = np.array(
            [
                quad_state.velocity.linear.x,
                quad_state.velocity.linear.y,
                quad_state.velocity.linear.z,
            ],
            dtype=np.float32,
        )
        self.omega = np.array(
            [
                quad_state.velocity.angular.x,
                quad_state.velocity.angular.y,
                quad_state.velocity.angular.z,
            ],
            dtype=np.float32,
        )

    def __repr__(self):
        repr_str = (
            "AgileQuadState:\n"
            + " t:     [%.2f]\n" % self.t
            + " pos:   [%.2f, %.2f, %.2f]\n" % (self.pos[0], self.pos[1], self.pos[2])
            + " att:   [%.2f, %.2f, %.2f, %.2f]\n"
            % (self.att[0], self.att[1], self.att[2], self.att[3])
            + " vel:   [%.2f, %.2f, %.2f]\n" % (self.vel[0], self.vel[1], self.vel[2])
            + " omega: [%.2f, %.2f, %.2f]"
            % (self.omega[0], self.omega[1], self.omega[2])
        )
        return repr_str
