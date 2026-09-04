import numpy as np
import xml.etree.ElementTree as ET

from robosuite.environments.manipulation.lift import Lift
from robosuite.utils.observables import Observable, sensor


class ConstrainedPickPlace(Lift):
    """
    Lift environment + fixed tall barrier + visible target marker.

    Barrier convention:
        barrier_pos       = center position in world coordinates
        barrier_half_size = MuJoCo box half extents [x, y, z]

    target_pos represents the desired final CUBE CENTER position.
    """

    def __init__(
        self,
        barrier_pos=(0.0, 0.04, 0.92),
        barrier_half_size=(0.12, 0.015, 0.12),
        table_z=0.8,
        *args,
        **kwargs,
    ):
        # These must be defined BEFORE super().__init__()
        # because Lift.__init__() eventually calls _load_model().
        self.barrier_pos = np.array(
            barrier_pos,
            dtype=np.float64,
        )

        self.barrier_half_size = np.array(
            barrier_half_size,
            dtype=np.float64,
        )

        self.table_z = float(table_z)

        # Desired final cube-center position.
        # Will be changed for every episode.
        self.target_pos = np.array(
            [0.0, 0.18, self.table_z + 0.03],
            dtype=np.float64,
        )

        super().__init__(*args, **kwargs)

    # ----------------------------------------------------------
    # Build MuJoCo scene
    # ----------------------------------------------------------

    def _load_model(self):
        # Let Lift create:
        # robot
        # table
        # cube
        # placement machinery
        super()._load_model()

        # ------------------------------------------------------
        # Physical barrier
        # ------------------------------------------------------

        barrier_body = ET.Element(
            "body",
            attrib={
                "name": "transport_barrier",
                "pos": self._vec_to_str(self.barrier_pos),
            },
        )

        ET.SubElement(
            barrier_body,
            "geom",
            attrib={
                "name": "transport_barrier_geom",
                "type": "box",
                "size": self._vec_to_str(
                    self.barrier_half_size
                ),
                "rgba": "0.25 0.25 0.25 1",
                # Important: make it visible
                "group": "1",
                "contype": "1",
                "conaffinity": "1",
                "friction": "1 0.005 0.0001",
            },
        )

        self.model.worldbody.append(barrier_body)

        # ------------------------------------------------------
        # Visual target marker
        # ------------------------------------------------------
        #
        # Very thin green box on table.
        # contype=0 / conaffinity=0 => visual only.
        #

        marker_pos = np.array(
            [
                self.target_pos[0],
                self.target_pos[1],
                self.table_z + 0.002,
            ]
        )

        target_body = ET.Element(
            "body",
            attrib={
                "name": "target_marker",
                "pos": self._vec_to_str(marker_pos),
            },
        )

        ET.SubElement(
            target_body,
            "geom",
            attrib={
                "name": "target_marker_geom",
                "type": "box",
                "size": "0.045 0.045 0.002",
                "rgba": "0.1 0.8 0.1 0.7",
                # Visible
                "group": "1",
                "contype": "0",
                "conaffinity": "0",
            },
        )

        self.model.worldbody.append(target_body)

    # ----------------------------------------------------------
    # Additional observations
    # ----------------------------------------------------------

    def _setup_observables(self):
        observables = super()._setup_observables()

        @sensor(modality="object")
        def target_pos(obs_cache):
            return self.target_pos.astype(np.float32)

        @sensor(modality="object")
        def barrier_pos(obs_cache):
            return self.barrier_pos.astype(np.float32)

        @sensor(modality="object")
        def barrier_half_size(obs_cache):
            return self.barrier_half_size.astype(np.float32)

        observables["target_pos"] = Observable(
            name="target_pos",
            sensor=target_pos,
            sampling_rate=self.control_freq,
        )

        observables["barrier_pos"] = Observable(
            name="barrier_pos",
            sensor=barrier_pos,
            sampling_rate=self.control_freq,
        )

        observables["barrier_half_size"] = Observable(
            name="barrier_half_size",
            sensor=barrier_half_size,
            sampling_rate=self.control_freq,
        )

        return observables

    # ----------------------------------------------------------
    # Change target during an episode
    # ----------------------------------------------------------

    def set_target(self, target_cube_pos):
        """
        target_cube_pos = desired final cube-center position.

        Moves the visible target marker to the same x/y location.
        """

        self.target_pos = np.asarray(
            target_cube_pos,
            dtype=np.float64,
        ).copy()

        body_id = self.sim.model.body_name2id(
            "target_marker"
        )

        self.sim.model.body_pos[body_id] = np.array(
            [
                self.target_pos[0],
                self.target_pos[1],
                self.table_z + 0.002,
            ],
            dtype=np.float64,
        )

        self.sim.forward()

    @staticmethod
    def _vec_to_str(v):
        return " ".join(str(float(x)) for x in v)