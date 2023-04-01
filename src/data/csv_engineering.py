

@csv_preprocessing_decorator
def separate_input_data_target(data, target_name):
    input_data = data.drop(['oc'], axis=1)
    target = data[target_name]
    return input_data, target


@csv_preprocessing_decorator
def separate_train_validate_test(data, frac_train=.6, frac_validate=.8):
    train, validate, test = np.split(data.sample(frac=1, random_state=42),
                                     [int(frac_train * len(data)), int(frac_validate * len(data))]
                                     )
    return train, validate, test


def to_numpy(data):
    return data.to_numpy()
