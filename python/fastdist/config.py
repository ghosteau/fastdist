import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pynvml

# ----------------------------------------------------------------------------------------------------------------------
# Constants and Defaults
# ----------------------------------------------------------------------------------------------------------------------

DEFAULT_CUDA_THRESHOLD = 100_000

_DEFAULT_CUDA_THRESHOLDS = {
    "normal": {
        "pdf": DEFAULT_CUDA_THRESHOLD,
        "logpdf": DEFAULT_CUDA_THRESHOLD,
        "cdf": DEFAULT_CUDA_THRESHOLD,
        "mgf": DEFAULT_CUDA_THRESHOLD,
        "cgf": DEFAULT_CUDA_THRESHOLD
    },
    "bernoulli": {
        "pmf": DEFAULT_CUDA_THRESHOLD,
        "cdf": DEFAULT_CUDA_THRESHOLD,
        "mgf": DEFAULT_CUDA_THRESHOLD,
        "cgf": DEFAULT_CUDA_THRESHOLD
    },
    "exponential": {
        "pdf": DEFAULT_CUDA_THRESHOLD,
        "cdf": DEFAULT_CUDA_THRESHOLD,
        "mgf": DEFAULT_CUDA_THRESHOLD,
        "cgf": DEFAULT_CUDA_THRESHOLD
    },
    "poisson": {
        "pmf": DEFAULT_CUDA_THRESHOLD,
        "cdf": DEFAULT_CUDA_THRESHOLD,
        "mgf": DEFAULT_CUDA_THRESHOLD,
        "cgf": DEFAULT_CUDA_THRESHOLD
    },
    "uniform": {
        "pdf": DEFAULT_CUDA_THRESHOLD,
        "cdf": DEFAULT_CUDA_THRESHOLD,
        "mgf": DEFAULT_CUDA_THRESHOLD,
        "cgf": DEFAULT_CUDA_THRESHOLD
    },
}

_DIST_CLASS_MAP = None

_FUNCTION_REGISTRY = None

_TESTING_PARAMETERS = {
    "normal": lambda: (0, 1),
    "bernoulli": lambda: (0.5,),
    "exponential": lambda: (10,),
    "poisson": lambda: (10,),
    "uniform": lambda: (1, 2)
}

_DEFAULT_SPACE_ARRAY = [10_000, 25_000, 50_000,
                        100_000, 250_000, 500_000,
                        1_000_000, 5_000_000, 10_000_000,
                        20_000_000, 25_000_000, 50_000_000,
                        100_000_000]

_ALLOWED_FUNCTIONS = {
    "normal": {"pdf", "logpdf", "cdf", "mgf", "cgf"},
    "bernoulli": {"pmf", "cdf", "mgf", "cgf"},
    "exponential": {"pdf", "cdf", "mgf", "cgf"},
    "poisson": {"pmf", "cdf", "mgf", "cgf"},
    "uniform": {"pdf", "cdf", "mgf", "cgf"},
}

# ----------------------------------------------------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------------------------------------------------

CUDA_THRESHOLDS = copy.deepcopy(_DEFAULT_CUDA_THRESHOLDS)

# ----------------------------------------------------------------------------------------------------------------------
# Configuration Path
# ----------------------------------------------------------------------------------------------------------------------

if platform.system() == "Windows":
    CONFIG_FILE = Path.home() / "AppData" / "Local" / "fastdist" / "config.json"
else:
    CONFIG_FILE = Path.home() / ".config" / "fastdist" / "config.json"

CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_config():
    """
    Load the CUDA threshold configuration from disk.

    This function attempts to read the JSON configuration file located at `CONFIG_FILE`.
    If the file exists and is valid, it updates the global `CUDA_THRESHOLDS` dictionary
    with the stored thresholds. If the file is missing or corrupted, it initializes
    `CUDA_THRESHOLDS` with the default values and writes a fresh configuration file.

    Side Effects:
        - Modifies the global variable `CUDA_THRESHOLDS`.
        - May create or overwrite the configuration file on disk.

    Raises:
        JSONDecodeError: If the file exists but is not valid JSON (handled internally).
    """

    global CUDA_THRESHOLDS
    CUDA_THRESHOLDS = copy.deepcopy(_DEFAULT_CUDA_THRESHOLDS)

    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            for key, value in data.items():
                if key in CUDA_THRESHOLDS and isinstance(value, dict):
                    CUDA_THRESHOLDS[key].update(value)
                else:
                    CUDA_THRESHOLDS[key] = value
            return
        except json.JSONDecodeError:
            pass

    # Corrupt or missing file, create default file
    CONFIG_FILE.write_text(json.dumps(CUDA_THRESHOLDS, indent=4))


def _save_config():
    """
    Persist the current CUDA thresholds to disk.

    Writes the global `CUDA_THRESHOLDS` dictionary to the configuration file at
    `CONFIG_FILE` in JSON format. Existing files are overwritten.

    Side Effects:
        - Writes to the configuration file.
    """

    CONFIG_FILE.write_text(json.dumps(CUDA_THRESHOLDS, indent=4))


# ----------------------------------------------------------------------------------------------------------------------
# Internal Lookup Helpers
# ----------------------------------------------------------------------------------------------------------------------

def _safe_get(obj, name):
    """
    Safely retrieve an attribute from an object.

    Parameters
    ----------
    obj : object
        The object from which to retrieve the attribute.
    name : str
        The name of the attribute to retrieve.

    Returns
    -------
    Any
        The value of the attribute if it exists; otherwise, None.
    """

    return getattr(obj, name, None)


def _get_dist_class_map():
    """Lazy-load the distribution class map to avoid circular imports."""
    global _DIST_CLASS_MAP

    if _DIST_CLASS_MAP is None:
        from fastdist import distributions as dists
        _DIST_CLASS_MAP = {
            "normal": dists.Normal,
            "poisson": dists.Poisson,
            "bernoulli": dists.Bernoulli,
            "exponential": dists.Exponential,
            "uniform": dists.Uniform,
        }

    return _DIST_CLASS_MAP


def get_function_pair(fd_function):
    global _FUNCTION_REGISTRY

    # Delay the import until the function is actually called
    if _FUNCTION_REGISTRY is None:
        # Now we hard code it inside the function
        _FUNCTION_REGISTRY = {
            "normal_pdf": ("_pdf_cpu", "_pdf_cuda"),
            "normal_logpdf": ("_logpdf_cpu", "_logpdf_cuda"),
            "normal_cdf": ("_cdf_cpu", "_cdf_cuda"),
            "normal_mgf": ("_mgf_cpu", "_mgf_cuda"),
            "normal_cgf": ("_cgf_cpu", "_cgf_cuda"),

            "poisson_pmf": ("_pmf_cpu", "_pmf_cuda"),
            "poisson_cdf": ("_cdf_cpu", "_cdf_cuda"),
            "poisson_mgf": ("_mgf_cpu", "_mgf_cuda"),
            "poisson_cgf": ("_cgf_cpu", "_cgf_cuda"),

            "bernoulli_pmf": ("_pmf_cpu", "_pmf_cuda"),
            "bernoulli_cdf": ("_cdf_cpu", "_cdf_cuda"),
            "bernoulli_mgf": ("_mgf_cpu", "_mgf_cuda"),
            "bernoulli_cgf": ("_cgf_cpu", "_cgf_cuda"),

            "exponential_pdf": ("_pdf_cpu", "_pdf_cuda"),
            "exponential_cdf": ("_cdf_cpu", "_cdf_cuda"),
            "exponential_mgf": ("_mgf_cpu", "_mgf_cuda"),
            "exponential_cgf": ("_cgf_cpu", "_cgf_cuda"),

            "uniform_pdf": ("_pdf_cpu", "_pdf_cuda"),
            "uniform_cdf": ("_cdf_cpu", "_cdf_cuda"),
            "uniform_mgf": ("_mgf_cpu", "_mgf_cuda"),
            "uniform_cgf": ("_cgf_cpu", "_cgf_cuda"),

            "sigmoid": ("sigmoid_cpu", "sigmoid_cuda"),
            "logit": ("logit_cpu", "logit_cuda"),
        }

    if fd_function not in _FUNCTION_REGISTRY:
        raise KeyError(f"Function {fd_function} not found in registry.")

    return _FUNCTION_REGISTRY[fd_function]


# ----------------------------------------------------------------------------------------------------------------------
# Benchmark Core
# ----------------------------------------------------------------------------------------------------------------------

def _benchmark(fd_function: str, display: int = 0, *args) -> int:
    """
    Determine the optimal CUDA threshold for a given distribution function.

    This function benchmarks the CPU and CUDA implementations of a FastDist function
    to find the smallest input size at which the CUDA implementation becomes faster
    than the CPU. It iteratively adjusts the threshold based on performance until
    convergence or a boundary is reached.

    Parameters
    ----------
    fd_function : str
        The function name in the format "<class>_<function>" corresponding to an entry
        in `_FUNCTION_MAP`.
    display : int, optional
        Level of console output: 0 = silent, 1 = summary, 2 = detailed debug (default 0).
    *args : tuple
        Arguments to pass to the distribution function during benchmarking.

    Returns
    -------
    int
        The CUDA threshold (array size) at which the CUDA implementation is faster
        than the CPU implementation.

    Raises
    ------
    RuntimeError
        If the CUDA implementation is not available for the given function or
        if the benchmark does not converge within the test iterations.
    """

    threshold_iter = 2  # Defaults to 100_000 starting point

    validation_reps = 10
    previous_sign = None
    cpu_name, cuda_name = get_function_pair(fd_function)

    class_name, _ = fd_function.split("_", 1)
    dist_cls = _get_dist_class_map()[class_name]
    instance = dist_cls(*args)

    cpu_func = getattr(instance, cpu_name)
    cuda_func = getattr(instance, cuda_name)

    if cuda_name is None:
        raise RuntimeError(f"CUDA not available for {fd_function}")

    if display > 1:
        print(f"""
        \t\t\t--- Benchmarking Settings ---
        \t\tStarting threshold_iter: {threshold_iter}
        \t\tStarting threshold value: {_DEFAULT_SPACE_ARRAY[threshold_iter]}
        \t\tValidation Repetitions: {validation_reps}
        \t\tFunctions to Benchmark:
        \t\t\t- CPU  : {cpu_name}
        \t\t\t- CUDA : {cuda_name}
        """)

    test_length = 50
    for _ in range(test_length):
        test_arguments = (_generate_int_array(_DEFAULT_SPACE_ARRAY[threshold_iter]), *args)
        print(test_arguments)

        if display > 1: print(f"\tTest {_} with array size {_DEFAULT_SPACE_ARRAY[threshold_iter]}")

        avg_cuda_time = _func_loop(cuda_func, validation_reps, *test_arguments)
        avg_cpu_time = _func_loop(cpu_func, validation_reps, *test_arguments)

        if display > 1: print(f"\tAverage CPU Time: {avg_cpu_time}\n\tAverage CUDA Time: {avg_cuda_time}")

        diff = avg_cuda_time - avg_cpu_time
        sign = diff > 0  # True = CUDA slower, False = CUDA faster

        if display > 1: print(f"\tDifference: {diff}\n\tSign: {sign}")

        if previous_sign is not None and sign != previous_sign:
            if display > 1: print(f"\tCrossed +/- boundary, exiting benchmark...")
            break  # Crossed +/- boundary

        previous_sign = sign

        if sign:
            if threshold_iter < len(_DEFAULT_SPACE_ARRAY) - 1:
                threshold_iter += 1
                if display > 1:
                    print(f"\tCUDA is slower, increasing the threshold to {_DEFAULT_SPACE_ARRAY[threshold_iter]}")
            else:
                if display > 1:
                    print(f"\tReached maximum threshold {_DEFAULT_SPACE_ARRAY[threshold_iter]}, stopping...")
                break  # At upper bound
        else:
            if threshold_iter > 0:
                threshold_iter -= 1
                if display > 1:
                    print(f"\tCUDA is faster, decreasing the threshold to {_DEFAULT_SPACE_ARRAY[threshold_iter]}")
            else:
                if display > 1:
                    print(f"\tReached minimum threshold {_DEFAULT_SPACE_ARRAY[threshold_iter]}, stopping...")
                break  # At lower bound
    else:
        raise RuntimeError("CUDA threshold benchmark did not converge")

    return _DEFAULT_SPACE_ARRAY[threshold_iter]


def _generate_int_array(number: int, low: int = 0, high: int = 10):
    """
    Generate a random integer array for benchmarking.

    Creates a 1-dimensional NumPy array of random integers, which is used as input
    for distribution function benchmarks.

    Parameters
    ----------
    number : int
        The number of integers to generate.
    low : int, optional
        The minimum integer value (inclusive, default 0).
    high : int, optional
        The maximum integer value (exclusive, default 10).

    Returns
    -------
    numpy.ndarray
        A 1-D array of random integers of length `number`.
    """

    return np.random.randint(low=low, high=high, size=number)


def _func_loop(function, repetitions, *args) -> float:
    """
    Measure the average execution time of a function over multiple iterations.

    Calls the provided function repeatedly with the given arguments and returns
    the average execution time in nanoseconds.

    Parameters
    ----------
    function : callable
        The function to execute.
    repetitions : int
        The number of times to call the function.
    *args : tuple
        Arguments to pass to the function during each call.

    Returns
    -------
    float
        The average execution time per call in nanoseconds.
    """

    total = 0
    for _ in range(repetitions):
        start_time = time.perf_counter_ns()

        function(*args)

        total += time.perf_counter_ns() - start_time
    return total / repetitions


# ----------------------------------------------------------------------------------------------------------------------
# String Utilities
# ----------------------------------------------------------------------------------------------------------------------
def merge_name_and_class(fd_class: str, fastdist_subfunction: str) -> str:
    """
    Combine a distribution class name and subfunction name into a single string.

    This utility is used to create keys for `_FUNCTION_MAP` or to
    reference a specific function in the format "<class>_<function>".

    Parameters
    ----------
    fd_class : str
        The name of the distribution class (e.g., "normal").
    fastdist_subfunction : str
        The name of the function (e.g., "pdf", "cdf").

    Returns
    -------
    str
        A string in the format "<class>_<function>".
    """

    return fd_class + '_' + fastdist_subfunction


def split_name_and_class(func: str) -> tuple[str, str]:
    """
    Split a combined function name into its class and subfunction components.

    Parameters
    ----------
    func : str
        The combined function name in the format "<class>_<function>".

    Returns
    -------
    tuple[str, str]
        A tuple `(fd_class, fastdist_subfunction)`.

    Raises
    ------
    ValueError
        If `func` does not contain an underscore or cannot be split into two parts.
    """

    if "_" not in func:
        raise ValueError(f"Function name '{func}' must be in the format <class>_<function>")
    fd_class, fastdist_subfunction = func.split("_", 1)
    return fd_class, fastdist_subfunction


# ----------------------------------------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------------------------------------
def auto_tune(classes: list[str] | None = None, functions: list[str] | None = None, display: int = 0):
    """
    Automatically benchmark and tune CUDA thresholds for FastDist functions.

    This function benchmarks the CPU and CUDA implementations of specified distribution
    functions or classes to determine the array size threshold where CUDA becomes
    faster than CPU. The results are saved to the configuration file.

    Parameters
    ----------
    classes : list[str] | None, optional
        List of distribution class names to benchmark (e.g., ["normal", "bernoulli"]).
        If None, classes are not benchmarked.
    functions : list[str] | None, optional
        List of specific functions to benchmark in the format "<class>_<function>".
        If None, individual functions are not benchmarked.
    display : int, optional
        Console output verbosity:
            0 - silent
            1 - summary
            2 - detailed debug (default 0)

    Raises
    ------
    ValueError
        If neither `classes` nor `functions` is provided, or if provided names are invalid.
    """

    if classes is None and functions is None:
        raise ValueError("Must provide at least one of `classes` or `functions`")

    if display > 0: print("--- Starting benchmarking ---")
    if classes is not None:
        for fd_class in classes:
            if fd_class not in CUDA_THRESHOLDS:
                raise ValueError(f"\tClass {fd_class} not found in CUDA_THRESHOLDS")
            if display > 0: print(f"\tBenchmarking class {fd_class}")
            for fd_function in CUDA_THRESHOLDS[fd_class]:
                test_arguments = _TESTING_PARAMETERS[fd_class]()
                if display > 0: print(f"\t- Function: {fd_function}{test_arguments}")
                CUDA_THRESHOLDS[fd_class][fd_function] = _benchmark(merge_name_and_class(fd_class, fd_function),
                                                                    display, *test_arguments)

    if functions is not None:
        for func in functions:
            fd_class, fd_function = split_name_and_class(func)
            if fd_class not in CUDA_THRESHOLDS:
                raise ValueError(f"\tClass {fd_class} not found in CUDA_THRESHOLDS")
            if fd_function not in CUDA_THRESHOLDS[fd_class]:
                raise ValueError(f"\tFunction {fd_function} not found in class {fd_class}")
            if classes is not None and fd_class in classes:
                if display > 0: print(f"\tFunction {func} has already been processed in class {fd_class}")
                continue  # Continue to the next function since the class benchmarks have already been run

            test_arguments = _TESTING_PARAMETERS[fd_class]()
            if display > 0: print(f"\tBenchmarking function {func}")
            CUDA_THRESHOLDS[fd_class][fd_function] = _benchmark(merge_name_and_class(fd_class, fd_function), display,
                                                                *test_arguments)

    if display > 0: print("Saving benchmark data...")
    _save_config()

    if display > 0: print(f"--- Benchmarking finished ---")


def get_cuda_threshold(func_name: str) -> int:
    """
    Retrieve the currently stored CUDA threshold for a FastDist function.

    Parameters
    ----------
    func_name : str
        The function name in the format "<class>_<function>".

    Returns
    -------
    int
        The CUDA threshold value for the specified function.

    Raises
    ------
    ValueError
        If `func_name` is empty or invalid.
    """

    if not func_name:
        raise ValueError("Must pass the function name as <class>_<function>")

    fd_class, fd_func = split_name_and_class(func_name)
    return CUDA_THRESHOLDS[fd_class][fd_func]


def set_cuda_threshold(func_name: str, value: int) -> None:
    """
    Manually set the CUDA threshold for a FastDist function.

    Updates the global `CUDA_THRESHOLDS` dictionary and optionally persists
    the new value via `_save_config()` if called afterwards.

    Parameters
    ----------
    func_name : str
        The function name in the format "<class>_<function>".
    value : int
        The new CUDA threshold value. Must be a positive integer.

    Raises
    ------
    ValueError
        If `func_name` is invalid, the class or function is not found,
        or if `value` is not a positive number.
    """

    if not func_name:
        raise ValueError("Must pass the function name as <class>_<function>")

    fd_class, fd_func = split_name_and_class(func_name)

    if fd_class not in CUDA_THRESHOLDS:
        raise ValueError(f"Class {fd_class} was not found")
    if fd_func not in CUDA_THRESHOLDS[fd_class]:
        raise ValueError(f"Function {fd_func} was not found in class {fd_class}")
    if value <= 0:
        raise ValueError("CUDA Threshold value must be a positive number")

    CUDA_THRESHOLDS[fd_class][fd_func] = value


def validate_gpu_capacity(array_size: int, dtype_item_size: int):
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        required = array_size * dtype_item_size * 2  # Factor of 2 for input and output arrays

        if required > info.free:
            raise MemoryError(
                f"GPU Memory Overflow: Required {required / 1e6:.2f}MB, "
                f"but only {info.free / 1e6:.2f}MB is free."
            )
    except pynvml.NVMLError:
        pass
    finally:
        pynvml.nvmlShutdown()


# ----------------------------------------------------------------------------------------------------------------------
# Module Initialization
# ----------------------------------------------------------------------------------------------------------------------
_load_config()
