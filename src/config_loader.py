import habitat
from habitat.config import Config
from habitat.sims import make_sim


def main():
    # setup environment
    config_path = (
        "configs/pointnav_rgbd.yaml"
    )
    config = habitat.get_config(config_path)
    # print(config)
    # env = habitat.PhysicsEnv(config=config)
    for k in config.PHYSICS_SIMULATOR.keys():
        if isinstance(config.PHYSICS_SIMULATOR[k], Config):
            for inner_k in config.PHYSICS_SIMULATOR[k].keys():
                config.SIMULATOR[k][inner_k] = config.PHYSICS_SIMULATOR[k][
                    inner_k
                ]
        else:
            config.SIMULATOR[k] = config.PHYSICS_SIMULATOR[k]

    try:
        from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
        from classes.habitat_physics_simulator import HabitatPhysicsSim
        from habitat.sims.habitat_simulator.actions import (
            HabitatSimV1ActionSpaceConfiguration,
        )
    except ImportError as e:
        print("Import HSIM failed")
        raise e
    print(config)
    sim = make_sim(
        id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR
    )
    # sim = registry.get_simulator(config.PHYSICS_SIMULATOR.TYPE)
    # print(sim)
    # print(env._sim)
    # print(env._config)


if __name__ == "__main__":
    main()
