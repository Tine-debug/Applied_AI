import numpy as np

dictionary = np.load("./Part1_warmup/vecs.npy", allow_pickle=True).item()
# dict_keys(['1_pos', '2_pos', '3_pos', '4_pos', '5_pos', '6_pos', '7_pos', '8_pos', '9_pos'])
# dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

def create_arrays(dictionary: dict, position: int) -> tuple[np.ndarray, np.ndarray]:
    if (position < 1 or position > 9):
        raise ValueError(f"{position} is not a valid position.")
    data = []
    labels = []
    for i in range (0, 10):
        for element in dictionary[f'{position}_pos'][i]:
            data.append(element)
            labels.append(i)
    return np.array(data), np.array(labels)

data, lables = create_arrays(dictionary, 1)
print(data.shape)
print(lables.shape)
        
def shuffle (data: np.array, lables: np.array) -> tuple[np.array, np.array]:
    indices = np.random.permutation(lables.shape[0])
    return data[indices], lables[indices]

Test_data = np.array([[1,1,2],[2,1,2],[3,1,2],[4,1,2],[5,1,2]])
Test_label = np.array([1,2,3,4,5])

Test_data, Test_label = shuffle(Test_data, Test_label)
print("shuffeld data: \n", Test_data)
print("shuffeld labels:", Test_label)
