import numpy as np

def constrained_pick_place_expert_action(obs, state, target_cube_pos, ctx):
    """
    target_cube_pos:
        desired final CENTER position of cube, shape (3,)

    ctx:
        dictionary storing persistent expert state such as grasp_offset
    """

    eef_pos = obs["robot0_eef_pos"]
    cube_pos = obs["cube_pos"]

    action = np.zeros(7, dtype=np.float32)

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
            ctx["grasp_count"] += 1
        else:
            ctx["grasp_count"] = 0

        # Hold grasp for ~0.5 sec at 10 Hz
        if ctx["grasp_count"] >= 5:
            ctx["grasp_offset"] = (
                eef_pos - cube_pos
            )

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
        route = ctx["route"]
        stage = ctx["transport_stage"]

        barrier_pos = obs["barrier_pos"]
        barrier_half_size = obs["barrier_half_size"]

        barrier_x = barrier_pos[0]
        barrier_y = barrier_pos[1]
        barrier_half_x = barrier_half_size[0]
        barrier_half_y = barrier_half_size[1]

        pre_y = (
            barrier_y
            - barrier_half_y
            - 0.08
        )

        post_y = (
            barrier_y
            + barrier_half_y
            + 0.08
        )


        # Enough clearance for cube + gripper
        clearance = 0.10

        if route == "LEFT":
            route_x = barrier_x - barrier_half_x - clearance
        else:
            route_x = barrier_x + barrier_half_x + clearance

        transport_z = target_cube_pos[2] + 0.15

        # Stage 0: move sideways before reaching barrier
        if stage == 0:
            desired_cube_pos = np.array([
                route_x,
                pre_y,
                transport_z,
            ])

            target_eef = desired_cube_pos + grasp_offset

            if abs(cube_pos[0] - route_x) < 0.015:
                ctx["transport_stage"] = 1

        # Stage 1: move forward beyond barrier
        elif stage == 1:
            desired_cube_pos = np.array([
                route_x,
                post_y,
                transport_z,
            ])

            target_eef = desired_cube_pos + grasp_offset

            if cube_pos[1] > barrier_y + 0.06:
                ctx["transport_stage"] = 2

        # Stage 2: move back toward target
        else:
            desired_cube_pos = target_cube_pos.copy()
            desired_cube_pos[2] = transport_z

            # Use current offset for better alignment
            current_offset = eef_pos - cube_pos
            target_eef = desired_cube_pos + current_offset

            xy_error = np.linalg.norm(
                cube_pos[:2] - target_cube_pos[:2]
            )

            if xy_error < 0.005:
                state = "LOWER"

        gripper_cmd = 1.0

    # ---------------------------------------------------------
    # 6. Lower cube onto table
    # ---------------------------------------------------------
    elif state == "LOWER":
        # Use current offset to compensate for any drift during transport
        current_offset = eef_pos - cube_pos

        target_eef = target_cube_pos + current_offset
        gripper_cmd = 1.0

        xy_error = np.linalg.norm(cube_pos[:2] - target_cube_pos[:2])
        z_error = abs(cube_pos[2] - target_cube_pos[2])

        if z_error < 0.012 and xy_error < 0.008:
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

    if state == "TRANSPORT":
        kp = 2.0
    elif state == "LOWER":
        kp = 4.0
    else:
        kp = 5.0
    

    action[:3] = np.clip(
        kp * error,
        -1.0,
        1.0,
    )

    action[6] = gripper_cmd

    return action, state