```python
from FeatureExtract import Feature Extract
```
```python
fe = FeatureExtract()
x, y = fe.load_data('path/to/gtzan_wavs', 'path/to/gtzan_csv')
```
Subsequent importing of the data can be made quicker with
```python
fe.save_csv('features.csv', features, labels)
x, y = fe.load_csv('features.csv')
```
If you want to tweak parameters in the code, afterwards
```python
fe.train(features, labels, predict=False)
```
To save and load the model after training
```python
fe.load_model(path)
fe.save_model(path=None, overwrite=False)
```
