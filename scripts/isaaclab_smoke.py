import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab smoke test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    sim = SimulationContext(SimulationCfg(dt=0.01))
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    sim.reset()
    for _ in range(10):
        sim.step()
    print("[SMOKE] Isaac Lab reset and 10 sim steps completed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
