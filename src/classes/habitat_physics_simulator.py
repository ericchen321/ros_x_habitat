import habitat_sim
from typing import (
    Any,
    List,
    Optional,
    Set,
)
import numpy as np
from gym import spaces
from classes.physics_simulator import PhysicsSimulator
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.spaces import Space
from habitat.core.simulator import (
    AgentState,
    Config,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    Sensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
    VisualObservation,
)


def overwrite_config_to_lower(
    config_from: Config, config_to: Any, ignore_keys: Optional[Set[str]] = None
) -> None:
    r"""Takes Habitat-API config and Habitat-Sim config structures. Overwrites
    Habitat-Sim config with Habitat-API values, where a field name is present
    in lowercase. Mostly used to avoid :ref:`sim_cfg.field = hapi_cfg.FIELD`
    code.

    Args:
        config_from: Habitat-API config node.
        config_to: Habitat-Sim config structure.
    """

    def if_config_to_lower(config):
        if isinstance(config, Config):
            return {key.lower(): val for key, val in config.items()}
        else:
            return config

    for attr, value in config_from.items():
        low_attr = attr.lower()
        if ignore_keys is None or low_attr not in ignore_keys:
            if hasattr(config_to, low_attr):
                setattr(config_to, low_attr, if_config_to_lower(value))
            else:
                raise NameError(
                    f"""{low_attr} is not found on habitat_sim but is found on habitat_lab config.
                    It's also not in the list of keys to ignore: {ignore_keys}
                    Did you make a typo in the config?
                    If not the version of Habitat Sim may not be compatible with Habitat Lab version: {config_from}
                    """
                )


@registry.register_simulator(name="Sim-v2")
class HabitatPhysicsSim(Simulator):
    r"""Simulator wrapper over habitat-sim

    habitat-sim repo: https://github.com/facebookresearch/habitat-sim

    Args:
        config: configuration for initializing the simulator.
    """

    def __init__(self, config: Config) -> None:
        self.habitat_config = config
        agent_config = self._get_agent_config()

        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.habitat_config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene_id
        self._sim = PhysicsSimulator(self.sim_config)
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        self._prev_sim_obs = None

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # Check if Habitat-Sim is post Scene Config Update
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )
        overwrite_config_to_lower(
            config_from=self.habitat_config.HABITAT_SIM_V0,
            config_to=sim_config,
            # Ignore key as it gets propogated to sensor below
            ignore_keys={"gpu_gpu"},
        )
        sim_config.scene_id = self.habitat_config.SCENE
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config_to_lower(
            config_from=self._get_agent_config(),
            config_to=agent_config,
            # These keys are only used by Hab-Lab
            ignore_keys={
                "is_set_start_state",
                # This is the Sensor Config. Unpacked below
                "sensors",
                "start_position",
                "start_rotation",
            },
        )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            sim_sensor_cfg = habitat_sim.SensorSpec()
            # TODO Handle configs for custom VisualSensors that might need
            # their own ignore_keys. Maybe with special key / checking
            # SensorType
            overwrite_config_to_lower(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                # These keys are only used by Hab-Lab
                # or translated into the sensor config manually
                ignore_keys={
                    "height",
                    "hfov",
                    "max_depth",
                    "min_depth",
                    "normalize_depth",
                    "type",
                    "width",
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(sensor.observation_space.shape[:2])
            sim_sensor_cfg.parameters["hfov"] = str(sensor.config.HFOV)

            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            # We know that the Sensor has to be one of these Sensors
            # sensor = cast(HabitatSimVizSensors, sensor)
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = self.habitat_config.HABITAT_SIM_V0.GPU_GPU
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.habitat_config.ACTION_SPACE_CONFIG
        )(self.habitat_config).get()

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def action_space(self) -> Space:
        return self._action_space

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.habitat_config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True

        return is_updated

    def reset(self):
        sim_obs = self._sim.reset()
        if self._update_agents_state():
            sim_obs = self._sim.get_sensor_observations()

        self._prev_sim_obs = sim_obs
        return self._sensor_suite.get_observations(sim_obs)

    def step(self, action):
        sim_obs = self._sim.step(action)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        return observations

    def step_physics(self, time_step):
        sim_obs = self._sim.step_physics(time_step)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        return observations

    def get_object_velocity_control(self, id_agent_obj):
        return self._sim.get_object_velocity_control(id_agent_obj)

    def render(self, mode: str = "rgb") -> Any:
        r"""
        Args:
            mode: sensor whose observation is used for returning the frame,
                eg: "rgb", "depth", "semantic"

        Returns:
            rendered frame according to the mode
        """
        sim_obs = self._sim.get_sensor_observations()
        observations = self._sensor_suite.get_observations(sim_obs)

        output = observations.get(mode)
        assert output is not None, "mode {} sensor is not active".format(mode)
        if not isinstance(output, np.ndarray):
            # If it is not a numpy array, it is a torch tensor
            # The function expects the result to be a numpy array
            output = output.to("cpu").numpy()

        return output

    def seed(self, seed):
        self._sim.seed(seed)

    def reconfigure(self, config: Config) -> None:
        # TODO(maksymets): Switch to Habitat-Sim more efficient caching
        is_same_scene = config.SCENE == self._current_scene
        self.habitat_config = config
        self.sim_config = self.create_sim_config(self._sensor_suite)
        if not is_same_scene:
            self._current_scene = config.SCENE
            self._sim.close()
            del self._sim
            self._sim = habitat_sim.Simulator(self.sim_config)

        self._update_agents_state()

    def geodesic_distance(
        self, position_a, position_b, episode: Optional[Episode] = None
    ):
        if episode is None or episode._shortest_path_cache is None:
            path = habitat_sim.MultiGoalShortestPath()
            if isinstance(position_b[0], List) or isinstance(position_b[0], np.ndarray):
                path.requested_ends = np.array(position_b, dtype=np.float32)
            else:
                path.requested_ends = np.array([np.array(position_b, dtype=np.float32)])
        else:
            path = episode._shortest_path_cache

        path.requested_start = np.array(position_a, dtype=np.float32)

        self._sim.pathfinder.find_path(path)

        if episode is not None:
            episode._shortest_path_cache = path

        return path.geodesic_distance

    def action_space_shortest_path(
        self, source: AgentState, targets: List[AgentState], agent_id: int = 0
    ) -> List[ShortestPathPoint]:
        r"""
        Returns:
            List of agent states and actions along the shortest path from
            source to the nearest target (both included). If one of the
            target(s) is identical to the source, a list containing only
            one node with the identical agent state is returned. Returns
            an empty list in case none of the targets are reachable from
            the source. For the last item in the returned list the action
            will be None.
        """
        raise NotImplementedError(
            "This function is no longer implemented. Please use the greedy "
            "follower instead"
        )

    @property
    def up_vector(self):
        return np.array([0.0, 1.0, 0.0])

    @property
    def forward_vector(self):
        return -np.array([0.0, 0.0, 1.0])

    def get_straight_shortest_path_points(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = position_a
        path.requested_end = position_b
        self._sim.pathfinder.find_path(path)
        return path.points

    def sample_navigable_point(self):
        return self._sim.pathfinder.get_random_navigable_point().tolist()

    def is_navigable(self, point: List[float]):
        return self._sim.pathfinder.is_navigable(point)

    def semantic_annotations(self):
        r"""
        Returns:
            SemanticScene which is a three level hierarchy of semantic
            annotations for the current scene. Specifically this method
            returns a SemanticScene which contains a list of SemanticLevel's
            where each SemanticLevel contains a list of SemanticRegion's where
            each SemanticRegion contains a list of SemanticObject's.

            SemanticScene has attributes: aabb(axis-aligned bounding box) which
            has attributes aabb.center and aabb.sizes which are 3d vectors,
            categories, levels, objects, regions.

            SemanticLevel has attributes: id, aabb, objects and regions.

            SemanticRegion has attributes: id, level, aabb, category (to get
            name of category use category.name()) and objects.

            SemanticObject has attributes: id, region, aabb, obb (oriented
            bounding box) and category.

            SemanticScene contains List[SemanticLevels]
            SemanticLevel contains List[SemanticRegion]
            SemanticRegion contains List[SemanticObject]

            Example to loop through in a hierarchical fashion:
            for level in semantic_scene.levels:
                for region in level.regions:
                    for obj in region.objects:
        """
        return self._sim.semantic_scene

    def close(self):
        self._sim.close()

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.habitat_config.DEFAULT_AGENT_ID
        agent_name = self.habitat_config.AGENTS[agent_id]
        agent_config = getattr(self.habitat_config, agent_name)
        return agent_config

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        assert agent_id == 0, "No support of multi agent in {} yet.".format(
            self.__class__.__name__
        )
        return self._sim.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        r"""Sets agent state similar to initialize_agent, but without agents
        creation. On failure to place the agent in the proper position, it is
        moved back to its previous pose.

        Args:
            position: list containing 3 entries for (x, y, z).
            rotation: list with 4 entries for (x, y, z, w) elements of unit
                quaternion (versor) representing agent 3D orientation,
                (https://en.wikipedia.org/wiki/Versor)
            agent_id: int identification of agent from multiagent setup.
            reset_sensors: bool for if sensor changes (e.g. tilt) should be
                reset).

        Returns:
            True if the set was successful else moves the agent back to its
            original pose and returns false.
        """
        agent = self._sim.get_agent(agent_id)
        original_state = self.get_agent_state(agent_id)
        new_state = self.get_agent_state(agent_id)
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = dict()
        agent.set_state(new_state, reset_sensors)
        return True

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(position, rotation, reset_sensors=False)

        if success:
            sim_obs = self._sim.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    def distance_to_closest_obstacle(self, position, max_search_radius=2.0):
        return self._sim.pathfinder.distance_to_closest_obstacle(
            position, max_search_radius
        )

    def island_radius(self, position):
        return self._sim.pathfinder.island_radius(position)

    @property
    def previous_step_collided(self):
        r"""Whether or not the previous step resulted in a collision

        Returns:
            bool: True if the previous step resulted in a collision, false otherwise

        Warning:
            This feild is only updated when :meth:`step`, :meth:`reset`, or :meth:`get_observations_at` are
            called.  It does not update when the agent is moved to a new loction.  Furthermore, it
            will _always_ be false after :meth:`reset` or :meth:`get_observations_at` as neither of those
            result in an action (step) being taken.
        """
        return self._prev_sim_obs.get("collided", False)
