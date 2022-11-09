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
chmod +x compile.sh
./compile.sh
```
