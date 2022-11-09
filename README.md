# Python/C API
This section describes how to call functions defined in Python and C respectively.

Newton's method will also be discussed as an implementation example.

## Preparation
First, run the following command.


```sh
# install pipenv
pip install pipenv
# setup PATH (option)
export PATH=$(echo ${HOME}/.local/bin):${PATH}
# create environment
pipenv --python 3.9
# install packages
pipenv install
```

Next, type the command and compile C module.

```
pipenv shell

cd c-api

chmod +x compile.sh
./compile.sh

cd -
```

## Example
Run the following command, then you will get sample outputs.

```sh
pipenv shell

# execute original script
python original/newton_method.py
# execute customized script with c library
python c-api/newton-method-with-clib.py
```
