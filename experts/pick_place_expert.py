import numpy as np

def pick_place_expert_action(obs, state, target_cube_pos, ctx):
    """
    target_cube_pos:
        desired final CENTER position of cube, shape (3,)

    ctx:
        dictionary storing persistent expert state such as grasp_offset
    """

    eef_pos = obs["robot0_eef_pos"]
    cube_pos = obs["cube_pos"]

    action = np.zeros(7, dtype=np.float32)

    kp = 5.0

    grasped = ctx["grasped"]

    # ---------------------------------------------------------
    # 1. Move above cube
    # ---------------------------------------------------------
    if state == "APPROACH":
        target_eef = cube_pos + np.array([0.0, 0.0, 0.05])
        gripper_cmd = -1.0

        xy_error = np.linalg.norm(eef_pos[:2] - cube_pos[:2])

        if xy_error < 0.008:
            state = "DESCEND"

    # ---------------------------------------------------------
    # 2. Descend to grasp pose
    # ---------------------------------------------------------
    elif state == "DESCEND":
        target_eef = cube_pos + np.array([0.0, 0.0, 0.015])
        gripper_cmd = -1.0

        z_error = eef_pos[2] - cube_pos[2]

        if z_error < 0.020:
            state = "GRASP"

    # ---------------------------------------------------------
    # 3. Close gripper
    # ---------------------------------------------------------
    elif state == "GRASP":
        target_eef = eef_pos.copy()
        gripper_cmd = 1.0

        if grasped:
            # Important:
            # Preserve geometric relation between EEF and cube.
            ctx["grasp_offset"] = eef_pos - cube_pos
            state = "LIFT"

    # ---------------------------------------------------------
    # 4. Lift vertically
    # ---------------------------------------------------------
    elif state == "LIFT":
        grasp_offset = ctx["grasp_offset"]

        desired_cube_pos = cube_pos.copy()
        desired_cube_pos[2] = target_cube_pos[2] + 0.12

        target_eef = desired_cube_pos + grasp_offset
        gripper_cmd = 1.0

        if cube_pos[2] > target_cube_pos[2] + 0.10:
            state = "TRANSPORT"

    # ---------------------------------------------------------
    # 5. Move above target while keeping cube high
    # ---------------------------------------------------------
    elif state == "TRANSPORT":
        grasp_offset = ctx["grasp_offset"]

        desired_cube_pos = target_cube_pos.copy()
        desired_cube_pos[2] += 0.12

        target_eef = desired_cube_pos + grasp_offset
        gripper_cmd = 1.0

        xy_error = np.linalg.norm(
            cube_pos[:2] - target_cube_pos[:2]
        )

        if xy_error < 0.012:
            state = "LOWER"

    # ---------------------------------------------------------
    # 6. Lower cube onto table
    # ---------------------------------------------------------
    elif state == "LOWER":
        grasp_offset = ctx["grasp_offset"]

        target_eef = target_cube_pos + grasp_offset
        gripper_cmd = 1.0

        z_error = abs(cube_pos[2] - target_cube_pos[2])

        if z_error < 0.012:
            state = "RELEASE"

    # ---------------------------------------------------------
    # 7. Release
    # ---------------------------------------------------------
    elif state == "RELEASE":
        target_eef = eef_pos.copy()
        gripper_cmd = -1.0

        if not grasped:
            state = "DONE"

    # ---------------------------------------------------------
    # Finished
    # ---------------------------------------------------------
    else:
        target_eef = eef_pos.copy()
        gripper_cmd = -1.0

    error = target_eef - eef_pos

    action[:3] = np.clip(
        kp * error,
        -1.0,
        1.0,
    )

    action[6] = gripper_cmd

    return action, state