from enum import Enum, IntEnum


PACKAGE_NAME = "ros_x_habitat"


class AgentResetCommands(IntEnum):
    RESET = 0
    SHUTDOWN = 1


class EvalEpisodeSpecialIDs(str, Enum):
    REQUEST_NEXT = "-1"
    REQUEST_SHUTDOWN = "-2"
    RESPONSE_NO_MORE_EPISODES = "-1"


class NumericalMetrics(str, Enum):
    DISTANCE_TO_GOAL = "distance_to_goal"
    SUCCESS = "success"
    SPL = "spl"
    NUM_STEPS = "num_steps"
    SIM_TIME = "sim_time"
    RESET_TIME = "reset_time"
    AGENT_TIME = "agent_time"


class ServiceNames(str, Enum):
    EVAL_EPISODE = "eval_episode"
    GET_AGENT_TIME = "get_agent_time"
    RESET_AGENT = "reset_agent"
    ROAM = "roam"
    GET_AGENT_POSE = "get_agent_pose"
