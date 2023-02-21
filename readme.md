

# Requirements

## C++
Create a folder in `cpp_src` called `.libs`. 

### Torch 
Download torch binaries from [here](https://pytorch.org/) (around 2 GB) and extract them into `.libs`.

### MariaDB
We need the C++ connector for MariaDB. Obtain both the C++ and C connector [here](https://mariadb.com/downloads/connectors/connectors-data-access/cpp-connector/). Extract the C connector into `.libs/mariadb_conn_c`, and the C++ connector into `.libs/mariadb_conn_cpp`.

### GSL
Install the *GNU Scientific Library* (GSL). The library must be put in the `.libs/gsl` directory. Follow instructions from [here](https://coral.ise.lehigh.edu/jild13/2016/07/11/hello/).


## Database
We use a MariaDB database. Using docker, create the container with

```
docker run --name alpha_db -p 127.0.0.1:3306:3306 -e MARIADB_USER=user  -e MARIADB_PASSWORD=password -e MARIADB_ROOT_PASSWORD=password -d mariadb 
```

Then you can access it in a terminal with
``` 
docker exec -it alpha_db mariadb --user root -ppassword
```

