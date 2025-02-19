import time
import os
import sys
from datetime import datetime
from common import parser

def check_exit(status: int):
    if status != 0:
        sys.exit(status)

def measure_time(command):
    start = time.time()
    status = os.system(command)
    end = time.time()
    elapsed = end - start
    return status, elapsed

if __name__ == "__main__":
    parser.parse_args()  # make --help work
    dirname = os.path.dirname(os.path.abspath(__file__))
    generation_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    print("Starting stage 1...")
    status, elapsed = measure_time(f'python {os.path.join(dirname, "infer_stage1.py")} {" ".join(sys.argv[1:])} --generation_timestamp {generation_timestamp}')
    print(f"Stage 1 completed in {elapsed:.2f} seconds")
    check_exit(status)

    print("Starting stage 2...")
    status, elapsed = measure_time(f'python {os.path.join(dirname, "infer_stage2.py")} {" ".join(sys.argv[1:])} --generation_timestamp {generation_timestamp}')
    print(f"Stage 2 completed in {elapsed:.2f} seconds")
    check_exit(status)

    print("Starting postprocessing...")
    status, elapsed = measure_time(f'python {os.path.join(dirname, "infer_postprocess.py")} {" ".join(sys.argv[1:])} --generation_timestamp {generation_timestamp}')
    print(f"Postprocessing completed in {elapsed:.2f} seconds")
    check_exit(status)

