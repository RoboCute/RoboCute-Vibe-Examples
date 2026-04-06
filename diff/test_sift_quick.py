#!/usr/bin/env python
"""Quick test script for SIFT implementation."""

import sys
sys.path.insert(0, 'diff')

import numpy as np
from sift import SIFT, SIFTConfig, SIFTDiff, sift_diff

print('Testing SIFT implementation...')

# Test 1: Basic SIFT detection
print('\n1. Testing basic SIFT detection...')
config = SIFTConfig(n_octaves=2, n_scales_per_octave=2)
sift = SIFT(config)

# Create test image with corners/edges for better SIFT detection
image = np.zeros((128, 128), dtype=np.float64)
# Create checkerboard-like pattern with clear corners
for i in range(0, 128, 32):
    for j in range(0, 128, 32):
        if (i // 32 + j // 32) % 2 == 0:
            image[i:i+32, j:j+32] = 1.0
# Add some smaller features
image[48:80, 48:80] = 0.5  # Gray square in center

keypoints = sift.detect_and_compute(image)
print(f'   Detected {len(keypoints)} keypoints')
assert len(keypoints) > 0, 'Should detect at least one keypoint'
print('   [OK] Basic detection works')

# Test 2: Keypoint has descriptor
print('\n2. Testing descriptor computation...')
kp_with_desc = [kp for kp in keypoints if kp.descriptor is not None]
print(f'   {len(kp_with_desc)} keypoints have descriptors')
assert len(kp_with_desc) > 0, 'Should have descriptors'
print('   [OK] Descriptor computation works')

# Test 3: Image comparison
print('\n3. Testing image comparison...')
diff = SIFTDiff()
results = diff.compare(image, image)
print(f'   Match ratio: {results["match_ratio"]:.3f}')
print(f'   Matches: {results["n_matches"]}')
assert results['n_matches'] > 0, 'Should have matches for identical images'
print('   [OK] Image comparison works')

# Test 4: Change detection
print('\n4. Testing change detection...')
has_changes, results = diff.detect_changes(image, image, match_threshold=0.5)
print(f'   Has changes (same image): {has_changes}')
assert not has_changes, 'Identical images should not have changes'

# Different images
image2 = np.zeros((128, 128), dtype=np.float64)
image2[20:40, 20:40] = 1.0
image2[80:100, 80:100] = 1.0
has_changes, results = diff.detect_changes(image, image2, match_threshold=0.5)
print(f'   Has changes (different images): {has_changes}')
assert has_changes, 'Different images should have changes'
print('   [OK] Change detection works')

# Test 5: Convenience function
print('\n5. Testing convenience function...')
results = sift_diff(image, image, n_octaves=2)
print(f'   Results keys: {list(results.keys())}')
assert 'match_ratio' in results
assert 'n_matches' in results
print('   [OK] Convenience function works')

print('\n[OK] All tests passed!')
