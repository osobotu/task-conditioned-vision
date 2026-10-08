
"""Compatibility fix for MuJoCo joint enum comparisons."""

import mujoco
from robosuite.utils.binding_utils import MjModel


def joint_address(model, name, address_field, widths):
    joint_id = model.joint_name2id(name)

    # Convert NumPy scalar to ordinary Python integer.
    joint_type = int(model.jnt_type[joint_id])
    address = int(getattr(model, address_field)[joint_id])

    if joint_type not in widths:
        raise ValueError(f"Unsupported joint type: {joint_type}")

    size = widths[joint_type]

    if size == 1:
        return address

    return (address, address + size)


def get_joint_qpos_addr(self, name):
    widths = {
        int(mujoco.mjtJoint.mjJNT_FREE): 7,
        int(mujoco.mjtJoint.mjJNT_BALL): 4,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }
    return joint_address(self, name, "jnt_qposadr", widths)


def get_joint_qvel_addr(self, name):
    widths = {
        int(mujoco.mjtJoint.mjJNT_FREE): 6,
        int(mujoco.mjtJoint.mjJNT_BALL): 3,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }
    return joint_address(self, name, "jnt_dofadr", widths)


def apply_patch():
    MjModel.get_joint_qpos_addr = get_joint_qpos_addr
    MjModel.get_joint_qvel_addr = get_joint_qvel_addr
