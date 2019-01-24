wget https://github.com/nest/nest-simulator/archive/v2.16.0.tar.gz

tar -xzvf nest-simulator-2.16.0.tar.gz

mkdir nest-simulator-2.16.0-build

cd nest-simulator-2.16.0-build

cmake -DCMAKE_INSTALL_PREFIX:PATH=$HOME/nest-simulator-2.16.0-build $HOME/nest-simulator-2.16.0

make

make install

# Very important! This sets the pythonpath variable
source nest-simulator-2.16.0-build/bin/nest_vars.sh
