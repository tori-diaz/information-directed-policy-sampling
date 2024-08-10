from functools import wraps
import time
import logging


def normalize(input_list):
    """
    Create a scaled version of the input list such that the elements sum to 1.

    Parameters
    ----------
    input_list : list
        

    Returns
    -------
    list
        A list whose elements sum to 1. 
    """

    return [i / sum(input_list) for i in input_list]


def create_uniform_pmf(num_of_points):
    """
    Create a discrete uniform pmf with the specified number of points.

    Parameters
    ----------
    num_of_points : int
        Number of points in the domain.

    Returns
    -------
    list
        A discrete uniform pmf over num_of_points points.
    """

    return [1/num_of_points] * num_of_points


def remove_indices_from_list(list_to_be_modified, indices_to_remove):
    """
    Modify list_to_be_modified so that the indices in indices_to_remove are removed.

    Parameters
    ----------
    list_to_be_modified : list      
        List whose elements will be removed

    indices_to_remove : list containing ints
        List containing the indices that will be removed

    Returns
    -------
    None
    """

    # indices_to_remove must be in ascending order
    indices_to_remove.sort()

    list_size = len(list_to_be_modified)

    list_to_be_modified.reverse()

    for index in indices_to_remove:
        del list_to_be_modified[list_size - index - 1]

    list_to_be_modified.reverse()

    return None


def include_zeroes(list_to_be_modified, index_list):
    """
    Modify list_to_be_modified such that there are 0's at the indices in index_list. 

    Example 
        Input: list_to_be_modified = [10, 12, 13, 14, 16], index_list = [1, 5]
        Ouput:   [10, 0, 12, 13, 0, 16]

    Parameters
    ----------
    list_to_be_modified : list
        The list that will be modified
        
    index_list : list of ints
        A list of indices indicating where to insert 0's in list_to_be_modified

    Returns
    -------
    None
    """

    for i in index_list:

        list_to_be_modified.insert(i, 0)

    return None


def timeit(func):
    """
    Standard wrapper function used to time other functions.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        total_time = time.perf_counter() - start_time

        print(f'{func.__name__}{args} {kwargs} Took {total_time:.2f} seconds')

        logging.debug(f'{func.__name__}{args} {
                      kwargs} Took {total_time:.2f} seconds')
        return result

    return wrapper
