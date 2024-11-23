def average_nn_parameters(parameters):
    """
    Averages passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    new_params = {}
    for name in parameters[0].keys():
        new_params[name] = sum([param[name].data for param in parameters]) / len(parameters)

    return new_params

def average_filtering(parameters):
    """
    Averages passed parameters with filtering abnormal clients.

    :param parameters: nn model named parameters
    :type parameters: list
    """

    # Calculate the mean direction of the parameters
    mean_direction = average_nn_parameters(parameters)

    # Calculate the distance of each client from the mean direction
    distances = []
    for param in parameters:
        distance = 0
        for name in param.keys():
            distance += ((param[name].data - mean_direction[name]) ** 2).sum().item()
        distances.append(distance)

    # Find the indices of the two clients farthest from the mean direction
    farthest_indices = sorted(range(len(distances)), key=lambda i: distances[i], reverse=True)[:2]

    # Remove the two farthest clients and get the filtered average parameters
    filtered_parameters = [param for i, param in enumerate(parameters) if i not in farthest_indices]
    filtered_average_params = average_nn_parameters(filtered_parameters)
    return filtered_average_params



def average_normalizarion(parameters):
    """
    Averages passed parameters with normalization client parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    normalized_params = {}
    
    for name in parameters[0].keys():
        mean_param = sum([param[name].data for param in parameters]) / len(parameters)
        distances = [(param[name].data - mean_param).norm().item() for param in parameters]
        farthest_indices = sorted(range(len(distances)), key=lambda i: distances[i], reverse=True)[:2]
        filtered_parameters = [param for i, param in enumerate(parameters) if i not in farthest_indices]
        normalized_params[name] = sum([param[name].data for param in filtered_parameters]) / len(filtered_parameters)

    return normalized_params