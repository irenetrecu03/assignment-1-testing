# README – Subgroup 2

## Python Version
All experiments, models, and automated tests were developed and executed using **Python 3.11.8**.  
The complete list of required dependencies is provided in **`subgroup_2/requirements.txt`**.

## Running Our Models and Tests
The training procedure for both the **good model** and **bad model** is documented in:

**`subgroup_2/models.ipynb`**

To reproduce our testing results, the automated tests are located in:

- **Partition tests:** `subgroup_2/partition_tests.py`  
- **Metamorphic tests:** `subgroup_2/metamorphic_tests.py`


## Testing the Other Subgroup’s Models
We evaluated the models created by the other subgroup using our own test suite.  
This is documented in:

**`subgroup_2/testing.ipynb`**

This notebook loads their models, runs both partition and metamorphic tests, and reports the results.
