import numpy as np
from typing import Callable, Optional


class KeyFrame:
    """
    Represents a discrete point in a timeline with associated data and events.
    
    Attributes:
        time: The time position of this keyframe (float)
        data: numpy array storing keyframe data values
        event: Optional callback function executed when keyframe is reached
    """
    
    def __init__(
        self,
        time: float,
        data: np.ndarray,
        event: Optional[Callable[[], None]] = None
    ):
        """
        Initialize a KeyFrame.
        
        Args:
            time: The time position of this keyframe
            data: numpy array containing the keyframe data values
            event: Optional callback function to execute at this keyframe
        """
        self.time = float(time)
        self.data = np.asarray(data, dtype=np.float64)
        self.event = event
    
    def __repr__(self) -> str:
        return f"KeyFrame(time={self.time}, data={self.data}, event={self.event is not None})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KeyFrame):
            return NotImplemented
        return (
            self.time == other.time and
            np.array_equal(self.data, other.data) and
            self.event == other.event
        )
    
    def execute_event(self) -> None:
        """Execute the associated event callback if present."""
        if self.event is not None:
            self.event()
    
    def get_data(self) -> np.ndarray:
        """Return a copy of the keyframe data."""
        return self.data.copy()
    
    def set_data(self, data: np.ndarray) -> None:
        """Update the keyframe data."""
        self.data = np.asarray(data, dtype=np.float64)
