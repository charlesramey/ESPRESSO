import numpy as np
from ESPRESSO import ESPRESSO

def test_espresso():
    # Generate synthetic data
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    # 3 segments: freq change
    s1 = np.sin(2 * np.pi * 0.1 * t[:300])
    s2 = np.sin(2 * np.pi * 0.5 * t[300:700])
    s3 = np.sin(2 * np.pi * 0.1 * t[700:])

    data = np.concatenate([s1, s2, s3])

    # Add some noise
    data += np.random.normal(0, 0.1, data.shape)

    # 1 dimension
    data = data.reshape(1, -1)

    K = 3
    subsequence = 50
    chain_len = 3

    boundaries = ESPRESSO(K, data, subsequence, chain_len)
    print("Found boundaries:", boundaries)

    # Check if we found 2 boundaries
    if len(boundaries) != 2:
        print("FAIL: Expected 2 boundaries")
    else:
        print("SUCCESS: Found 2 boundaries")

if __name__ == "__main__":
    test_espresso()
