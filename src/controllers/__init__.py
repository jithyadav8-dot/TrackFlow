from src.controllers.fcfs import FCFS
from src.controllers.longest_wait import LongestWaiting
from src.controllers.priority import FixedPriority
from src.controllers.work_conserving import WorkConserving

CONTROLLERS = {
    "FCFS": FCFS,
    "Fixed Priority": FixedPriority,
    "Longest Waiting": LongestWaiting,
    "Work Conserving": WorkConserving,
}
