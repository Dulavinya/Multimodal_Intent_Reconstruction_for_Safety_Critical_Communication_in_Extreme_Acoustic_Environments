"""
Command vocabulary for AthenAI.
Maps integer class indices to safety command strings.
Extend this list to match your actual training labels.
"""

COMMAND_VOCAB = [
    "stop the machine",
    "emergency shutdown",
    "evacuate immediately",
    "reduce speed",
    "increase pressure",
    "decrease pressure",
    "open valve",
    "close valve",
    "activate safety lock",
    "release safety lock",
    "call supervisor",
    "check sensor",
    "restart system",
    "halt conveyor",
    "start conveyor",
    "fire alarm",
    "chemical leak alert",
    "electrical hazard",
    "all clear",
    "unknown command",
]

# Inverse map: command string → index
COMMAND_TO_IDX = {cmd: idx for idx, cmd in enumerate(COMMAND_VOCAB)}
