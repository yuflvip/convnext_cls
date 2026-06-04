from clearml import Task
import time

task = Task.init(project_name="AIPlatform/YOLO", task_name="agent-queue-test")
logger = task.get_logger()

for i in range(5):
    print(f"queue step={i}")
    logger.report_scalar("queue_metric", "value", value=i, iteration=i)
    time.sleep(1)

print("queue test done")
