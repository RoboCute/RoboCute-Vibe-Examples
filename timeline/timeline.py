import numpy as np
from typing import List, Union, Optional

# Import from the other modules
try:
    from .keyframe import KeyFrame
    from .interpolation import Interpolation, InterpolationType
except ImportError:
    from keyframe import KeyFrame
    from interpolation import Interpolation, InterpolationType


class Timeline:
    """
    Manages a sequence of KeyFrame and Interpolation instances.
    
    Structure constraints:
    - First element must be KeyFrame
    - Last element must be KeyFrame
    - Interpolation must always be between two KeyFrame instances
    - Pattern: [KeyFrame, Interpolation, KeyFrame, Interpolation, KeyFrame, ...]
    
    Attributes:
        elements: List of KeyFrame and Interpolation instances
    """
    
    def __init__(self):
        """Initialize an empty Timeline."""
        self.elements: List[Union[KeyFrame, Interpolation]] = []
    
    def __repr__(self) -> str:
        num_keyframes = sum(1 for e in self.elements if isinstance(e, KeyFrame))
        num_interpolations = sum(1 for e in self.elements if isinstance(e, Interpolation))
        return f"Timeline(keyframes={num_keyframes}, interpolations={num_interpolations})"
    
    def __len__(self) -> int:
        """Return the number of elements in the timeline."""
        return len(self.elements)
    
    def _validate_structure(self) -> bool:
        """
        Validate the timeline structure.
        
        Returns:
            True if structure is valid
            
        Raises:
            ValueError: If structure is invalid
        """
        if len(self.elements) == 0:
            return True
        
        # First element must be KeyFrame
        if not isinstance(self.elements[0], KeyFrame):
            raise ValueError("First element must be KeyFrame")
        
        # Last element must be KeyFrame
        if not isinstance(self.elements[-1], KeyFrame):
            raise ValueError("Last element must be KeyFrame")
        
        # Check alternating pattern: KeyFrame, Interpolation, KeyFrame, ...
        for i, element in enumerate(self.elements):
            if i % 2 == 0:  # Even indices (0, 2, 4, ...) must be KeyFrame
                if not isinstance(element, KeyFrame):
                    raise ValueError(f"Element at index {i} must be KeyFrame")
            else:  # Odd indices (1, 3, 5, ...) must be Interpolation
                if not isinstance(element, Interpolation):
                    raise ValueError(f"Element at index {i} must be Interpolation")
        
        return True
    
    def add_keyframe(self, keyframe: KeyFrame) -> None:
        """
        Add a KeyFrame to the timeline.
        
        If timeline is empty or has odd length (ends with Interpolation), 
        adds KeyFrame directly.
        If timeline has even length (ends with KeyFrame), adds an Interpolation
        before adding the KeyFrame.
        
        Args:
            keyframe: The KeyFrame to add
        """
        if len(self.elements) == 0:
            self.elements.append(keyframe)
        elif len(self.elements) % 2 == 1:  # Ends with KeyFrame, need Interpolation
            # Add default linear interpolation
            self.elements.append(Interpolation(interp_type=InterpolationType.LINEAR))
            self.elements.append(keyframe)
        else:  # Ends with Interpolation, can add KeyFrame directly
            self.elements.append(keyframe)
        
        self._validate_structure()
    
    def add_interpolation(self, interpolation: Interpolation) -> None:
        """
        Add an Interpolation to the timeline.
        
        Can only add Interpolation if timeline ends with a KeyFrame
        and has at least one KeyFrame.
        
        Args:
            interpolation: The Interpolation to add
            
        Raises:
            ValueError: If interpolation cannot be added
        """
        if len(self.elements) == 0:
            raise ValueError("Cannot add Interpolation to empty timeline")
        
        if len(self.elements) % 2 == 0:  # Already ends with Interpolation
            raise ValueError("Cannot add Interpolation after Interpolation")
        
        self.elements.append(interpolation)
        # Note: Timeline is temporarily invalid here until next KeyFrame is added
        # Validation happens when add_keyframe is called
    
    def get_keyframes(self) -> List[KeyFrame]:
        """Return all KeyFrame instances in the timeline."""
        return [e for e in self.elements if isinstance(e, KeyFrame)]
    
    def get_interpolations(self) -> List[Interpolation]:
        """Return all Interpolation instances in the timeline."""
        return [e for e in self.elements if isinstance(e, Interpolation)]
    
    def evaluate(self, time: float) -> Optional[np.ndarray]:
        """
        Evaluate the timeline at a given time.
        
        Finds the appropriate KeyFrame or Interpolation segment for the given time
        and returns the interpolated data.
        
        Args:
            time: The time position to evaluate
            
        Returns:
            Interpolated data as numpy array, or None if timeline is empty
        """
        if len(self.elements) == 0:
            return None
        
        keyframes = self.get_keyframes()
        
        # Find the keyframe segment
        for i in range(len(keyframes) - 1):
            kf_start = keyframes[i]
            kf_end = keyframes[i + 1]
            
            if kf_start.time <= time <= kf_end.time:
                if time == kf_start.time:
                    return kf_start.get_data()
                if time == kf_end.time:
                    return kf_end.get_data()
                
                # Find the interpolation between these keyframes
                start_idx = self.elements.index(kf_start)
                end_idx = self.elements.index(kf_end)
                
                if end_idx - start_idx == 2:  # There's an interpolation
                    interp = self.elements[start_idx + 1]
                    # Calculate t parameter
                    duration = kf_end.time - kf_start.time
                    if duration > 0:
                        t = (time - kf_start.time) / duration
                    else:
                        t = 0.0
                    return interp.interpolate(kf_start.data, kf_end.data, t)
                else:
                    # No interpolation, use linear
                    duration = kf_end.time - kf_start.time
                    if duration > 0:
                        t = (time - kf_start.time) / duration
                    else:
                        t = 0.0
                    return kf_start.data + (kf_end.data - kf_start.data) * t
        
        # Time is before first keyframe or after last
        if time <= keyframes[0].time:
            return keyframes[0].get_data()
        return keyframes[-1].get_data()
    
    def sample(self, num_samples: int) -> np.ndarray:
        """
        Sample the timeline at regular intervals.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Array of shape (num_samples, data_dim) containing sampled values
        """
        if len(self.elements) == 0:
            return np.array([])
        
        keyframes = self.get_keyframes()
        if len(keyframes) == 0:
            return np.array([])
        
        start_time = keyframes[0].time
        end_time = keyframes[-1].time
        
        samples = []
        for i in range(num_samples):
            if num_samples > 1:
                t = i / (num_samples - 1)
            else:
                t = 0.0
            time = start_time + (end_time - start_time) * t
            sample = self.evaluate(time)
            if sample is not None:
                samples.append(sample)
        
        return np.array(samples)
