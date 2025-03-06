import numpy as np
import matplotlib.pyplot as plt
import sys
import random

# Hash table size
TABLE_SIZE = 10000
LOAD_FACTORS = np.linspace(0.1, 0.95, 10)  # Varying load factors
NUM_KEYS = [int(TABLE_SIZE * load) for load in LOAD_FACTORS]

# Hash table methods
METHODS = [
    # "Random Probing",
    # "Elastic Hashing",
    # "Funnel Hashing",
    "Bathroom Model"
]

# Results storage
average_lookup_time = {method: [] for method in METHODS}
worst_case_probes = {method: [] for method in METHODS}
memory_utilization = {method: [] for method in METHODS}
MAX_PROBES = 100  # Prevent infinite loops

def hash_function(key, size=TABLE_SIZE):
    """Simple hash function for simulation purposes"""
    return key % size

def get_memory_usage(table):
    """Estimate memory usage of the hash table (in bytes)"""
    return sys.getsizeof(table) + sum(sys.getsizeof(item) for item in table if item is not None)

def random_probing(table, key):
    """Traditional random probing method"""
    index = int(hash_function(key))
    probes = 1  # Start with first probe attempt
    while table[index] is not None and probes < MAX_PROBES:
        index = (index + np.random.randint(1, TABLE_SIZE)) % TABLE_SIZE
        probes += 1
    table[index] = key if table[index] is None else table[index]  # Avoid overwriting
    return probes

def elastic_hashing(table, key):
    """Elastic hashing method based on the paper"""
    index = int(hash_function(key))
    probes = 1  # Start with first probe attempt
    jump = 1
    while table[index] is not None and probes < MAX_PROBES:
        index = (index + jump) % TABLE_SIZE
        jump *= 2  # Exponential step increase
        probes += 1
    table[index] = key if table[index] is None else table[index]
    return probes

def funnel_hashing(table, key):
    """Funnel hashing method, dividing table into decreasing size regions"""
    index = int(hash_function(key))
    probes = 1  # Start with first probe attempt
    sub_table_size = TABLE_SIZE
    while table[index] is not None and probes < MAX_PROBES:
        sub_table_size = max(2, sub_table_size // 2)  # Ensure sub_table_size is at least 2
        index = (index + np.random.randint(1, sub_table_size)) % TABLE_SIZE
        probes += 1
    table[index] = key if table[index] is None else table[index]
    return probes

def bathroom_model(table, key):
    """Bathroom Model: Adaptive dynamic probing"""
    index = int(hash_function(key))
    probes = 1  # Start with first probe attempt
    step_size = 1  # Start with a small step
    while table[index] is not None and probes < MAX_PROBES:
        step_size = min(step_size * 2, TABLE_SIZE // 4)  # Adaptive step growth
        index = (index + step_size) % TABLE_SIZE
        probes += 1
    table[index] = key if table[index] is None else table[index]
    return probes



mbLookupAttempts = 0
def bmLookup(table, key):
    """Bathroom Model: Adaptive dynamic probing"""
    global mbLookupAttempts

    tableSize = len(table)
    index = int(hash_function(key, tableSize))
    probes = 1  # Start with first probe attempt
    step_size = 1  # Start with a small step
    while table[index] != key and probes < MAX_PROBES:
        step_size = min(step_size * 2, tableSize // 4)  # Adaptive step growth
        index = (index + step_size) % tableSize
        probes += 1
        mbLookupAttempts += 1

    if index == tableSize:
        return None, -1

    return table[index], index


def mbLookupTester(table, key):
    global mbLookupAttempts
    mbLookupAttempts = 0

    keyFound, indexFound = bmLookup(table, key)
    print(f"Lookup {key} : {keyFound} : index {indexFound} :: attempts {mbLookupAttempts}")


mbInsertAttempts = 0
def bmInsert(table, key):
    """Bathroom Model: Adaptive dynamic probing"""
    global mbInsertAttempts
    MAX_ATTEMPTS = 10

    tableSize = len(table)
    index = int(hash_function(key, tableSize))
    probes = 1  # Start with first probe attempt
    step_size = 1  # Start with a small step
    while table[index] is not None and probes < MAX_ATTEMPTS:
        step_size = min(step_size * 2, tableSize // 4)  # Adaptive step growth
        index = (index + step_size) % tableSize
        probes += 1
        mbInsertAttempts += 1

    if table[index] is not None:
        index = 0
        while table[index] is not None and index < tableSize:
            index += 1
            mbInsertAttempts += 1

    if index >= tableSize:
        return -1

    table[index] = key
    return index


def perf_test():
    # Running experiments
    for num_keys in NUM_KEYS:
        for method in METHODS:
            table = [None] * TABLE_SIZE
            probes_list = []
            keys = np.random.randint(1, 10**6, num_keys).tolist()  # Convert to list for integer indexing
            for key in keys:
                if method == "Random Probing":
                    probes_list.append(random_probing(table, key))
                elif method == "Elastic Hashing":
                    probes_list.append(elastic_hashing(table, key))
                elif method == "Funnel Hashing":
                    probes_list.append(funnel_hashing(table, key))
                elif method == "Bathroom Model":
                    probes_list.append(bathroom_model(table, key))

            average_lookup_time[method].append(np.mean(probes_list))
            worst_case_probes[method].append(np.max(probes_list))
            memory_utilization[method].append(get_memory_usage(table))  # Record memory usage

    # Plot results
    plt.figure(figsize=(10, 5))
    for method in METHODS:
        plt.plot(NUM_KEYS, average_lookup_time[method], label=method, marker='o')
    plt.xlabel("Number of Keys Inserted")
    plt.ylabel("Average Lookup Time")
    plt.title("Comparison of Hash Table Lookup Efficiency")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    for method in METHODS:
        plt.plot(NUM_KEYS, worst_case_probes[method], label=method, marker='o')
    plt.xlabel("Number of Keys Inserted")
    plt.ylabel("Worst-Case Probe Complexity")
    plt.title("Comparison of Worst-Case Probing")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    for method in METHODS:
        plt.plot(NUM_KEYS, memory_utilization[method], label=method, marker='o')
    plt.xlabel("Number of Keys Inserted")
    plt.ylabel("Memory Utilization (bytes)")
    plt.title("Comparison of Memory Utilization Across Methods")
    plt.legend()
    plt.grid()
    plt.show()


def ins_lookup_test():
    MY_TABLE_SIZE = 100
    table = [None] * MY_TABLE_SIZE

    for i in range(0, MY_TABLE_SIZE):
        j = random.randint(375482, 6729084)
        index = bmInsert(table, j)
        inserted = index != -1
        print("{} : index {}  --- {}".format(j, index, inserted))
    print(f"Insert attempts {mbInsertAttempts}")

    global mbLookupAttempts
    mbLookupAttempts = 0

    for i, key in enumerate(table):
        mbLookupTester(table, key)

ins_lookup_test()
