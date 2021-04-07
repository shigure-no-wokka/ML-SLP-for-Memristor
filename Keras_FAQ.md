# Keras中文学习文档笔记
## Keras FAQ:
### 1.如何引用 Keras ？
```python
'''
@misc{chollet2015keras,
	author = {Chollet, Franois, and others},
	title = {Keras},
	year = {2015},
	publisher = {GitHub},
	journal = {GitHub repository},
	howpublished = {\url{https://github.com/fchollet/keras}}
}
'''
```
### 2.‘batch’，‘epoch’，‘sample’都是什么意思？
- Sample：样本，是数据集中的一条数据。例如图片数据集中的一张图片
- **Batch**：一个 batch 由若干条数据构成。batch 是进行网络优化的基本单位，网络参数的每一轮优化都需要使用一个 batch。batch 中的样本是被并行处理的。与单个样本相比，**一个batch的数据能更好的模拟数据集的分布，batch 越大则对输入数据分布模拟的越好，反应在网络训练上，则体现为能让网络训练的方向“更加准确”**。但另一方面，一个 batch 也只能让网络的参数更新一次，因此网络参数的迭代会较慢。在测试网络的时候，应该在条件允许的范围内尽量使用更大的 batch，这样计算效率会比较高
- **Epoch**：epoch 可以翻译为“轮次”。如果每个 batch 对应网络的一次更新，那么一个 epoch 则对应于网络的一整轮更新。每一次更新中网络更新的次数可以随意，但是通常会设置为遍历一遍数据集。因此**一个 epoch 的含义是模型完整的看完了一次数据集**。设置 epoch 的主要作用是把模型的训练的整个训练过程分成若干个段，这样可以更好的观察和调整模型的训练。  

### 3.如何保存 Keras 模型？
不推荐使用 pickle 或cPickle 来保存。可以使用 model。save(filepath) 将 Keras 的模型和权重保存在同一个 HDF5 文件中，文件将包含：
- 模型的结构，以便重构该模型
- 模型的权重
- 训练配置（损失函数，优化器等）
- 优化器的状态，以便于从上次训练中断的地方开始  

使用 keras.models.load_model(filepath) 来重新实例化模型，如果文件中存储了训练配置，该函数还会同时完成模型的编译
例子：
```python
from keras.models import load_model
from keras.models import save_model

model = save_model('my_model.h5') # 建立一个 HDF5 文件
del model # deletes the existing model

# return a compiled model
# identical to the previous one
model = load_model('my_model.h5')
```
如果*只希望保存模型的结构，而不包含权重或配置信息*，可以使用：
```python
# save as JSON
json_string = model.to_json()

# save as YAML
yaml_string = model.to_yaml()
```
这些操作把模型保存为了 json 和 yaml文件，如果需要的话可以直接打开这些文件进行编辑  
也可以从 json 或 yaml 文件中载入模型：
```python
# model reconstruction from json
from keras.models import model_from_json, model_from_yaml
model = model_from_json(json_string=0)

# model reconstruction from YAML
model = model_from_yaml(yaml_string=0)
```
如果要*保存模型的权重*，可以通过下面的代码利用 HDF5 进行保存。注意在使用前，需要确保你已经安装了 HDF5 和其 Python 库 h5py
```python
model.save_weights('my_model_weights.h5')
```
如果需要在代码中初始化一个完全相同的模型，使用：
```python
model.load_weights('my_model_weights.h5')
```
如果需要加载权重到不同的网络模型中，可以通过层名字来加载模型：
```python
model.load_weights('my_model_weights.h5', by_name=True)
```
例如：
```python
# 假如原模型为：
    model_old = Sequential()
    model_old.add(Dense(2, input_dim=3, name='dense_1'))
    model_old.add(Dense(3, name='dense_2'))
    ...
    model_old.save_weights(fname)

# 新模型
    model_new = Sequential()
    model_new.add(Dense(2, input_dim=3, name='dense_1')) # 加载
    model_new.add(Dense(3, name='new_dense_2')) # 不会被加载
    ...
    # 从第一个模型中载入权重，制会影响第一层，dense_1
    model_new.load_weights(fname, by_name=True)
```
### 4.为什么训练误差比测试误差高很多？
一个 Keras 的模型有两个模式：训练模式和测试模式。  
训练误差是训练数据每个 batch 的误差的平均。在训练过程中，每个 epoch 起始时的 batch 的误差要大一些，
而后面的 batch 的误差要小一些。另一方面，每个 epoch 结束时计算的测试误差是由模型在 epoch 结束时的状态
决定的，这时候的网络将产生较小的误差。
【Tips】可以通过定义回调函数将每个 epoch 的训练误差和测试误差并作图，如果训练误差曲线和测试误差曲线
之间与很大的空隙，说明模型可能有过拟合的问题。
### 还有其他一些暂时未整理