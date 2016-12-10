+++
date = "2016-12-03T14:38:42+05:30"
next = "/next/path"
prev = "/notes/CNN-Karpathy-CS231-Stanford"
title = "Python Tricks"
toc = true
weight = 5

+++

*** 
### IPython Tricks
* Create matplotlib graphs inlined 
%matplotlib inline

* Display image and other formats in ipython notebook
https://ipython.org/ipython-doc/2/api/generated/IPython.core.display.html
e.g for image
~~~
display(Image(filename=imageName)
~~~

* Display images horizontally
~~~
import matplotlib.pyplot as plt
from PIL import Image
imgs = [] # Stores name of files  
f,ax = plt.subplots(1,5, figsize=(15,15))
for i in range(5):
    ax[i].imshow(Image.open(imgs[i]))
plt.show() # or display.display(plt.gcf()) if you prefer
~~~

* Display single image from buffer
~~~
pl.imshow(Image.fromarray(X)) 
~~~

* Setup Remote notebook server.
Only follow instructions on http://jupyter-notebook.readthedocs.io/en/latest/public_server.html


***
### TFLearn Tricks

{{% notice warning %}}
Sometimes while using TFLearn/Tensorflow with IPython notebook, the session/graph does not get free and the new runs keep reflecting old errors. Restart the kernel to remove such issue. This should be first step to debug persistent and unexplainable error. 
{{% /notice %}}

***
### Numpy Tricks
* Convert a ndarray of (num_features, image_size, image_size) to (num_features, image_size * image_size) of float32 type
~~~
dataset.reshape((-1, image_size * image_size)).astype(np.float32)
~~~

* Create a one hot encoding for features from range (0-num_labels) to vector of length num_length. Mapping value x -> vec of len num_length with xth value as 1.0.
Here labels is a nparray of feature vector of length (num_features) with range (0-num_labels) each.
~~~
(np.arange(num_labels) == labels[:,None]).astype(np.float32)
~~~

* Change size of imshow window and plotting downsampled array - a 
~~~
fig = plt.figure(figsize=(2, 2))
plt.imshow(asd[::2,::2])
~~~

***
### Python debugging

* To locate segmentation fault in python script, use following code to create call trace.
~~~
def trace(frame, event, arg):
    print "%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno)
    return trace
sys.settrace(trace)
~~~

* asd

### List and map enumerations

* Create a list of (index, x[0], x[1]) where x is each element of stateTable
~~~
list(enumerate((x[0],x[1]) for x in stateTable))
~~~

* Create a map from x[0] to (x[1],x[2]) 
~~~
{x[0]:(x[1],x[2]) for x in stateTable}
~~~

* Nested list comprehension. Equivalent for 
newList = []
for x in l:
  for y in x:
    newList.append(float(y))
~~~
[[float(y) for y in x] for x in l]
~~~

* List enumeration in map enumaration
~~~
{
  'asd1':[n for n in range(1,100) if n%9 == 0],
  'asd2':[n for n in range(1,100) if n%7 == 0]
}
~~~  

### Python functions

* Zip fucntion. zip function returns a list of tuples, where the i-th tuple contains the i-th element from each of the argument sequences or iterables. 
 x = [1, 2, 3]
 y = [4, 5, 6]
~~~
 zipped = zip(x, y) # Zips x and y lists to create a list (x,y) tuples
 x2, y2 = zip(*zipped) # Unzips to x2, y2 lists
~~~ 

* Use zip to create map enumeration
~~~
{number: letter for letter, number in zip('ankur',range(1,10))}
{1: 'a', 2: 'n', 3: 'k', 4: 'u', 5: 'r'}
~~~
 
### Python recipes

* Sort a dictionary by value ( returns a sorted list of (key, value) pairs
~~~
sorted(d.items(), key=lambda x: x[1])
~~~
or
~~~
import operator
sorted(d.items(), key=operator.itemgetter(1))
~~~

* Read lines from stdin
~~~
import sys
lines = sys.stdin.readlines()
~~~
