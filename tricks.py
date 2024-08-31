#-------------LIST INSERT----------------
a: list[str] = ['0', '1', '3']
a.insert(2, '2')
# [0, 1, '2', 3]

#-------------TUPLE UNPACK----------------
a = ('Joe')
type(a) 
# string

#-------------DECORATOR---------------
def my_decor(func):
	def inner(*args, **kwargs):
		'start'
		func(*args, **kwargs)
		'end'
		return True
	return inner

@my_decor
def calc(a, b):
	return a + b

calc(1, 4)
"""
start
5
end
-> True
"""

#-----------ASTERISK FOR VARIABLES-----------------
x = 1, 2, 3, 4
a, *b, d = x
#b == [2, 3]

#--------------GLOBAL-----------------
n = 10

def func():
	global n
	n += 10
	#n = 20
func()
#n == 20

#------------NONLOCAL------------------
def big_func():
	x = 1

	def small_func():
		nonlocal x
		x = 3

	small_func()
	#x == 3

big_func()

#------------ITERATOR-----------------
class MyTime(object):
	def __iter__(self):
		self.l = 0
		return self

	def __next__(self):
		self.l += 2
		return self.l

myiter = iter(MyTime()) # iterable object

for _ in range(10):
	next(myiter) # 0, 2, 4, 6, ..., 20

#-----------GENERATOR FROM FUNCTION-------------
from collections.abc import Callable
from typing import Iterable


def fib(lock: int):
	a, b = 0, 1

	while a < lock:
		yield a
		a, b = b, a + b

for i in fib(10):
	i # 0, 1, 1, 2, 3, 5, 8, ...

x: Iterable[int] = fib(10)
x.__next__() #0
x.__next__() #1

#----------------YIELD FROM-------------------
def it(array: Iterable[int]) -> int:
	yield from array

for el in it([1, 2, 3, 4]):
	el # 1 2 3 4

#-----------------COROUTINE-------------------
def len_checker(length: int) -> bool:
	try:
		while True:
			word: str = (yield)
			len(word) == length # print
	except GeneratorExit:
		'exit'

check = len_checker(3)
check.__next__()
check.send('abc')
check.send('abcde')
check.close()
"""
True
False
exit
"""

#------------TYPICAL GENERATOR-----------------
x = (i * i for i in range(10))
# x == generator object <genexpr>
for i in x:
	i # 0, 1, 4, ..., 81

#------------SORT----------------- 
def comp(x):
	return x[0]

x = [0, 2, 1, 4]
a = sorted(x)
# a == [0, 1, 2, 4]
t = [(0, 1), (2, '1'), (1, [3])]
t.sort(reverse=False, key=comp)
# t == [(0, 1), (1, [3]), (2, '1')]

#------------FROZEN SET-----------------
x = frozenset([1, 2, 3]) 
# immutable set

#------------METHODS--------------------
from abc import (
	ABC, abstractmethod, 
	abstractstaticmethod, abstractclassmethod
)
from typing import Literal, NoReturn


class Human(ABC):
	@abstractmethod
	def swim(self, style: Literal['butterfly'] | \
		Literal['dog']) -> NoReturn:
		pass

	@abstractstaticmethod
	def calculate(a: int, b: int) -> int:
		pass

	@abstractclassmethod
	def clone(cls):
		pass

class Swimmer(Human):
	def __init__(self, age: int = 0):
		self.age = age

	def grow(self, age: int) -> int:
		self.age += age

	@staticmethod
	def calculate(a: int, b: int) -> int:
		return a + b

	@classmethod
	def clone(cls) -> 'Swimmer':
		return cls(10)

	def swim(self, style: Literal['butterfly'] | \
		Literal['dog']) -> NoReturn:
		f'{self.age} y. o. and swim like a {style}'


h1 = Swimmer(20)
Swimmer.calculate(2, 2) # 4
h2 = Swimmer.clone() # basic swimmer with h2.age == 10
h1.grow(20) # h1.age == 40
Swimmer.clone().swim('dog') # 10 y. o. and swim like a dog

#------------GENERATORS ADVANCED------------
[i for i in range(10) if i >= 2 if i % 4 == 0 if i % 8 == 0] # [8] cases in generator
['a' if j > 3 else 'b' for j in range(15)] # if-clause

#--------------CHAIN SUM CHALLENGE------------
from typing import Optional


def chain_sum(num1: int) -> Callable[[Optional[int]], int | Callable]:
	def in_chain(num2: Optional[int] = None) -> int | Callable:
		return num1 if num2 is None else chain_sum(num1 + num2)
	return in_chain


chain_sum(4)(5)(-10)(30)(8)() # 37

#------------SECOND MAXIMUM AND TESTS------------
def second_max(array):
	if len(array) < 2:
		return None
	max1, max2 = max(array[0], array[1]), min(array[0], array[1])

	for num in array[2:]:
		if num > max1:
			max1, max2 = num, max1
		elif num > max2 and num != max1:
			max2 = num
	return max2 if max2 != max1 else None


def test_second_max():
	res = second_max([12, 50, 3, 4, 12, -100, 0, 50])
	assert res == 12, f'Wrong: {res}'
	res = second_max([])
	assert res == None, f'Wrong: {res}'
	res = second_max([1, 1])
	assert res == None, f'Wrong: {res}'
	res = second_max([3, 2, -10, 2, 100, 45])
	assert res == 45, f'Wrong: {res}'

	return 'OK'

test_second_max() # OK

#-------------UNIQUE ARRAY OF EVEN NUMBERS------------
from typing import Sequence


def unique_even(array: Sequence[int]) -> list[int]:
	return [
		array[i] for i in range(len(array))
		if not array[i] % 2 and 
		((i + 1 < len(array) and
		array[i] not in array[i + 1:]) or
		(i + 1 == len(array)))
	]

unique_even([2, 15, 3 , 9, 1, -10, 7]) # [2, -10]

#------------------DECIMAL-----------------------
x = 0.1
y = 0.1
z = 0.1
x + y + z # 0.30000000000000004

from decimal import Decimal

x: Decimal = Decimal('0.1')
y: Decimal = Decimal('0.1')
z: Decimal = Decimal('0.1')
x + y + z # 0.3

#----------------CUSTOM SEQUENCE---------------
from typing import NamedTuple, Sequence
from datetime import datetime

Days = int

class Food(NamedTuple):
	creation_time: datetime
	max_keep: Days

class FoodList:
	def __init__(self, foods: Sequence[Food]):
		self._foods = foods

	def __getitem__(self, ind: int) -> Food:
		return self._foods[ind]

foods = FoodList((
	Food(creation_time=datetime.fromisoformat('2022-01-21 05:44:10'), max_keep=10),
	Food(creation_time=datetime.fromisoformat('2022-02-12 06:14:36'), max_keep=7),
	Food(creation_time=datetime.fromisoformat('2022-03-30 15:36:50'), max_keep=160)
))
foods[0] # Food(creation_time=datetime.datetime(2022, 1, 21, 5, 44, 10), max_keep=10)

#------------------GENERICS------------------
from typing import TypeVar

T = TypeVar('T')

def first(iterable: Iterable[T]) -> T | None:
	for element in iterable:
		return element

first(['one', 'two']) # 'one'

#----------------REDUCE---------------
from functools import reduce

reduce(
	lambda x, y: x * y, 
	map(int, '10 3 32 25'.split())
) # 24000

#----------------FILTER---------------
array = ['Cat', 'Dog', 'Bat', 'Fox', 'Butterfly']
list(filter(
	lambda x: x.startswith('B'), 
	array
)) # ['Bat', 'Butterfly']

#-------------OPERATOR >> __rshift__-----------
115 >> 3 # 14 [1110011 -> ___1110]

#-------------THE WALRUS OPERATOR-----------
info = 'Hello, world'
if info_len := len(info):
	f'Info length is {info_len}' # Info length is 12


#----------------CLASS INSTANCE TO DICT---------------
class Some:
	def __init__(self):
		self.x = 1.1
		self.y = -2

vars(Some()) # {'x': 1.1, 'y': -2}

#------------CREATE CLASS WITHOUT CLASS---------------
CoolClass = type('CoolClass', (), {'value': 0})
cooler = CoolClass()
cooler.value # 0

#------------------METACLASS---------------------
class Meta(type):
	def __new__(self, class_name, base, attrs):
		a = dict()
		for name, val in attrs.items():
			if name.startswith('__') and name.endswith('__'):
				a[name] = val
			else:
				a[name.upper()] = val
		return type(class_name, base, a)

class What(metaclass=Meta):
	lat = 1
	long = 10

	def show(self):
		print('lol')

dir(What) # ['LAT', 'LONG', 'SHOW', '__class__', ...

#-------------TYPE OF TYPE IS TYPE--------------
type(type(type(type))) # <class 'type'>

#------------INTERFACE WITH PROTOCOL--------------
from typing import Protocol, runtime_checkable


@runtime_checkable
class Basic(Protocol):
	def show(self):
		...

class Advanced(Basic):
	pass

Advanced().show() # duck typying (nothing fails)

#----------PROXY FACTORY DESIGN PATTERN-----------
from abc import ABCMeta, abstractstaticmethod


class IPerson(metaclass=ABCMeta):
	@abstractstaticmethod
	def person_method():
		pass

class Person(IPerson):
	def person_method(self):
		print('I am a person')

class ProxyPerson(IPerson):
	def __init__(self):
		self.person = Person()

	def person_method(self):
		print('I am the proxy person')
		self.person.person_method()

#------------SINGLETON DESIGN PATTERN--------------
from abc import ABCMeta, abstractstaticmethod


class IPerson(metaclass=ABCMeta):
	@abstractstaticmethod
	def print_data():
		"""imlement in child"""

class PersonSingleton(IPerson):
	__instance = None

	def __init__(self, name: str, age: 0):
		if PersonSingleton.__instance is not None:
			raise Exception('Person was initiated before.')
		self.name = name
		self.age = age
		PersonSingleton.__instance = self

	@staticmethod
	def get_instance():
		if PersonSingleton.__instance is None:
			PersonSingleton('name', 0)
		return PersonSingleton.__instance

	@staticmethod
	def print_data():
		print(
			PersonSingleton.__instance.name,
			PersonSingleton.__instance.age
		)

p = PersonSingleton('Tom', 90)
p0 = PersonSingleton.get_instance()
p # <object at 0x7fd7abb07010>
p0 # <object at 0x7fd7abb07010>
# p.print_data() # Tom 90
# PersonSingleton('Joe', 54) # Exception: Person was initiated before.

#------------------------ ; ------------------------
def look(place: str):
	pass

look('here'); look('somewhere'); look('anywhere')

#--------------MATRIX TO VECTOR-------------
from itertools import chain
matr = (
	(0, 1),
	(1, 0, 1),
	(1, 1)
)
tuple(chain.from_iterable(matr))
# (0, 1, 1, 0, 1, 1, 1)

#------------------COUNTER--------------
from collections import Counter

counter = Counter('aaabcccc101')
# Counter({'c': 4, 'a': 3, '1': 2, 'b': 1, '0': 1})
counter.keys() # dict_keys(['a', 'b', 'c', '1', '0'])
counter.values() # dict_values([3, 1, 4, 2, 1])
counter.most_common(2) # [('c', 4), ('a', 3)]
list(counter.elements())
# ['a', 'a', 'a', 'b', 'c', 'c', 'c', 'c', '1', '1', '0']

#------------DEFAULT DICT--------------
from collections import defaultdict

d = defaultdict(int)
d.update({'a': 4, 'b': 5})
d['smth'] # 0

#--------------COMBINATIONS--------------
from itertools import product

list(product([1, 2], [3, 4]))
#(1, 3), (1, 4), (2, 3), (2, 4)]

from itertools import permutations

list(permutations((1, 2, 3), 3))
# [(1, 2, 3), (1, 3, 2), (2, 1, 3), 
# (2, 3, 1), (3, 1, 2), (3, 2, 1)]

from itertools import combinations

list(combinations([1, 2, 3], 2))
# [(1, 2), (1, 3), (2, 3)]

from itertools import accumulate
import operator

list(accumulate(
	[1, 2, 3, 4, 5], 
	func=operator.mul)
) # [1, 2, 6, 24, 120]

#--------------EXACT COPY--------------
from copy import deepcopy

array1 = [
	[0, 1, 0],
	[1, 0, 1],
	[0, 1, 0]
]
array2 = deepcopy(array1)
array2[0][0] += 1
# array1 is not changed