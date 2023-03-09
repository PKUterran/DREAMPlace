apt-get update
apt-get install libboost-all-dev -y
apt search boost -y

apt-get install bison
apt-get install flex -y

pip install seaborn
pip install -r requirements.txt 
pip install --pre dgl -f https://data.dgl.ai/wheels/cu113/repo.html
pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install pympler
pip install memory_profiler
mkdir model
mkdir result
mkdir -p log/pretrain
mkdir -p log/train

cd thirdparty/kahypar
mkdir build 
cd build
pwd
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make


cd ../../../
mkdir build
cd build
install_path=`pwd`
cmake .. -DCMAKE_INSTALL_PREFIX=$install_path -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_CUDA_FLAGS=-gencode=arch=compute_86,code=sm_86
make 
make install


