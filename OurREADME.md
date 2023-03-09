1.KaHyPar尽量用单核性能高的处理器跑，因为KaHyPar单线程，经测试在 AMD EPYC 7642 (2.3GHz)3h都跑不完superblue19但是在Intel Xeon Platinum 8358P(2.6GHz)上10min就能出结果


2.代码在https://github.com/PKUterran/DREAMPlace.git的huawei/code分支下，clone下来后执行
```
    git submodule update --init --recursive
```
来clone子模块


3.假如在clone子模块的时候出现以下问题
请按照如下步骤解决
找到最底层出现问题的子模块然后依次向上执行如下操作
```
    cd {submodule path}
    git reset --hard origin/master
    cd -
    git clean -n
    git add {submodule path}
    git commit -m "xxx"
```
直到最顶一层，即我们的DREAMPlace仓库，其中git status发现working tree是干净的后重新执行如下指令
```
    git submodule update --init --recursive
```
假如clone子模块的过程中很慢
一个加速思路是配置ssh，然后将较慢的子模块的url从HTTPS换成仓库内的SSH，然后在父模块中执行
```
    git submodule sync
```
再执行
```
    git submodule update --init --recursive
```


4.InstallDependency.sh目前是负责kahypar DREAMPlace mt-kahypar的编译工作和依赖包的下载
中间可能会有bug,如果出现问题,直接去对应的build目录下把指令重新执行一遍即可


5.我们的模型的评测代码在dreamplace/script_evaluate.py,代码里面有两个list,netlist_name中存储的是每个网表数据的路径,
默认放在benchmarks文件夹下面,param_json为对应的param.json路径，假如没有会自动生成默认的，生成对应的param.json的py为GenerateParamJson.py，里面有lef_input,def_input,verilog_input,根据数据可以修改默认生成的param.json格式
执行示例为(在DREAMPlace根目录下)
```
    python dreamplace/script_evaluate.py (--device cuda:0) (--model model) (--name default)
```

执行DREAMPlace的示例为(在DREAMPlace根目录下)
```
    python dreamplace/script_DREAMPlace.py
```

同样的DREAMPlace里面也有list，含义与上相同，netlist_name中存储的是每个网表数据的路径,
默认放在benchmarks文件夹下面,param_json为对应的param.json路径，假如没有会自动生成默认的

假如现在./DREAMPlace/benchmarks/ispd2019/ispd19_test1/ 下有对应的lef/def
那么在script_evaluate.py中的netlist_name和param_json为

```
NETLIST_DIR='benchmarks'
test_param_json_list = [
        'test/OurModel/ispd19_test1/ispd19_test1.json',
]
test_netlist_names = [
    f'{NETLIST_DIR}/ispd2019/ispd19_test1'
]
```
假设模型为model.pkl
则测试命令为
```
    python dreamplace/script_evaluate.py --device cuda:0 --model model
```
--name不填默认为default

在script_DREAMPlace.py中为

```
NETLIST_DIR='benchmarks'
test_param_json_list = [
        'test/DREAMPlace/ispd19_test1/ispd19_test1.json'
]
test_netlist_names = [
    f'{NETLIST_DIR}/ispd2019/ispd19_test1'
]
```
则测试命令为
```
    python dreamplace/script_DREAMPlace.py
```

6.目前默认用的是kahypar但是也可以用mt-kahypar，可以在grouping.py中找到修改地方(在create_group函数中)

7.现在模型放在https://github.com/sorarain/Model.git仓库下，clone下来后直接执行里面的tar.sh可以得到我们的模型model.pkl，放在DREAMPlace的model文件夹(InstallDependency.sh会创建这个文件夹)下即可

8.我们的最终布局在result/{args.name}/{netlist_name}/{netlist_name}_gp.def

9.DREAMPlace最终布局在result/DREAMPlace/{netlist_name}/DREAMPlace_{netlist_name}_gp.def