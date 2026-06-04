
from clearml import Task

task = Task.init(project_name="AIPlatform/YOLO", task_name="clearml-connect-test")
logger = task.get_logger()
logger.report_scalar("test", "value", value=1, iteration=0)
print("clearml connected")

