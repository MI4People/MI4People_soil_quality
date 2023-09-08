import numpy as np


def geospatial_split():
    pass


# Currently unused, see data.general_datapipes for alternative
def ttv_split(data, split_ratio=[0.7, 0.2, 0.1]):
    split_data = np.split(
        data.sample(frac=1, random_state=42),
        [
            int(split_ratio[0] * len(data)),
            int((split_ratio[0] + split_ratio[1] * len(data))),
        ],
    )
    return split_data
