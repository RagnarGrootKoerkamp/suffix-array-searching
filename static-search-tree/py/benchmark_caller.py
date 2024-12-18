import os
import subprocess
import shlex

def parse_output(cmd_output: str) -> list[tuple[int, float]]:
    lines = [line.split() for line in cmd_output.split("\n")]
    lines = list(filter(lambda line: len(line) == 2, lines))
    return [(int(arr[0]), float(arr[1])) for arr in lines]

def get_measurements(fname: str, start_pow2: int, stop_pow2: int, queries: int) -> list[tuple[int, float]]:
    project_path = os.path.join(os.path.dirname(__file__), "../")
    args = shlex.split(f"cargo run --bin bench_one -- --fname {fname} --start {start_pow2} --stop {stop_pow2} --queries {queries}")
    result = subprocess.check_output(args, cwd=project_path).decode()
    return parse_output(result)

if __name__ == "__main__":
    print(get_measurements("bs_search", 10, 20, 1000))
