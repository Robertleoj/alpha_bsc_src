

# Requirements

## C++


Create a folder in `cpp_src` called `.libs`. 

### Torch 
Download torch binaries from [here](https://pytorch.org/) (around 2 GB) and extract them into `.libs`.

### MariaDB
Install C++ connector from [here](https://mariadb.com/docs/skysql/connect/programming-languages/cpp/install/).

### GSL
Install the *GNU Scientific Library* (GSL). On ubuntu, this is 
```
sudo apt install libgsl-dev
```
On Fedora, use 
```
sudo dnf install gsl
```

## Database
We use a MariaDB database. Using docker, create the image with


```
docker run --name alpha_db -p 127.0.0.1:3306:3306 -e MARIADB_USER=user  -e MARIADB_PASSWORD=password -e MARIADB_ROOT_PASSWORD=password -d mariadb 
```

Then you can access it in a terminal with
``` 
docker exec -it alpha_db mariadb --user root -ppassword
```