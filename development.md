## Running unit tests

Native python:
```
python -m unittest tests/
```
### Installing test dependencies
Before running the unit tests, install the required packages:

```
pip install -r requirements_versions.txt
```

### Environment
Tests use the same configuration files as the application. You can override paths such as `config_path` or `config_example_path` using environment variables.


Embedded python (Windows zip file installation method):
```
..\python_embeded\python.exe -m unittest
```
