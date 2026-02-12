# This file defines any custom types or data structures used throughout the application, facilitating type checking and improving code clarity.

from typing import List, Dict, Any

# Example of a custom type for a data structure
class HealthData:
    def __init__(self, date: str, recovery_score: float, sleep_efficiency: float, activity_strain: float):
        self.date = date
        self.recovery_score = recovery_score
        self.sleep_efficiency = sleep_efficiency
        self.activity_strain = activity_strain

# Example of a type alias for a list of health data entries
HealthDataList = List[HealthData]

# Example of a configuration type
Config = Dict[str, Any]