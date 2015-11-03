# hybrid_image.py setup


Installing Python
=========
* Please download anaconda with Python 3 from: https://www.continuum.io/downloads/
* Install it according to your system using the instructions found on that website
* In your command prompt ensure you have downloaded anaconda by typing
```
conda --version

```

Running hybrid_image.py
===========

To run the function run_hybrid:
* Load your command prompt (cmd in Windows, terminal in Mac and Linux).
* Change the directory to the one where hybrid_image.py is located.
* Either make sure your images are also in this folder, or in the command
  give the full filepath as well as the image name.
* Type the following command into the command prompt:
```
python hybrid_image.py image1 image2
```
* ```image1``` and ```image2``` should be in the format cat.bmp or
  * Windows: C:\\Users\\username\\Documents\\hybrid_image\\cat.bmp
  * Mac or Linux: ~/hybrid_image/cat.bmp

Testing hybrid_image.py
===========

To test the whole hybrid_image module using the given test suite more
requirements are needed.

* Install nose by typing

```
conda install nose
```
* If this does not work on UNIX-like systems then try running using ```sudo```
* Change the directory to the test_hybrid_image.py and hybrid_image.py directory.
  These two files must be in the same folder
* Type the command
```
nosetests
```
* If any of the tests failed it will come back with the number of failed tests and what the errors were.
```
Ran 6 tests in 5.598s

FAILED (failures=1)
```
* Otherwise it will come back saying
```
Ran 5 tests in 6.082s

OK
```
