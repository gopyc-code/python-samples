from typing import (
	Sequence, Callable, TypeVar, NoReturn, Optional
)
from copy import deepcopy


comp = lambda a1, a2: a1 > a2

Item = TypeVar('Item')
Key = TypeVar('Key')

def sorter(
	sort_func: Callable[[Sequence[Item]], NoReturn],
	array: Sequence[Item]) -> Sequence[Item]:
	copied = deepcopy(array)
	sort_func(copied)
	return copied


def test_sorter(
	sort_func: Callable[[Sequence[Item]], NoReturn]) -> Optional[bool]:
	array = [289, 3, 2, 200, 3, -200, 15, 93, 1, 0, 8]
	assert sorter(sort_func, array) == [-200, 0, 1, 2, 3, 3, 8, 15, 93, 200, 289], \
	'Array must be [-200, 0, 1, 2, 3, 3, 8, 15, 93, 200, 289]'
	array = []
	assert sorter(sort_func, array) == [], 'Array must be empty'
	array = [-100, 0, 50, -100]
	assert sorter(sort_func, array) == [-100, -100, 0, 50], \
	'Array must be [-100, -100, 0, 50]' 
	array = [1]
	assert sorter(sort_func, array) == [1], 'Array must be [1]'
	array = [2, 1]
	assert sorter(sort_func, array) == [1, 2], 'Array must be [1, 2]'
	
	return True

#------------------------BUBBLE SORT-------------------------
def bubble_sort(array: Sequence[Item]) -> NoReturn:
	for i in range(len(array)):
		for j in range(len(array) - i - 1):
			if comp(array[j], array[j + 1]):
				array[j], array[j + 1] = array[j + 1], array[j]

#----------------------SELECTION SORT------------------------
def selection_sort(array: Sequence[Item]) -> NoReturn:
	for i in range(len(array)):
		min_ind = i
		for j in range(i + 1, len(array)):
			if comp(array[min_ind], array[j]):
				min_ind = j
		array[i], array[min_ind] = array[min_ind], array[i]


#----------------------INSERTION SORT------------------------
def insertion_sort(array: Sequence[Item]) -> NoReturn:
	for i in range(1, len(array)):
		value = array[i]
		j = i
		while j > 0 and comp(array[j - 1], value):
			array[j] = array[j - 1]
			j -= 1
		array[j] = value


#----------------------MERGE SORT------------------------
def merge_sort(array: Sequence[Item]) -> NoReturn:
	array_length = len(array)
	if array_length < 2: return
	left = array[:array_length // 2]
	right = array[array_length // 2:]

	merge_sort(left)
	merge_sort(right)

	for i in range(array_length):
		if left and right:
			if comp(right[0], left[0]):
				array[i] = left[0]
				left.pop(0)
			else:
				array[i] = right[0]
				right.pop(0)
	unused = left + right
	array[array_length - len(unused):] = unused


#----------------------QUICK SORT----------------------
def quick_sort_recursive(
	array: Sequence[Item], 
	L: int, R: int) -> NoReturn:
	if len(array) < 2 or L >= R: return

	value, pointer = array[R], L
	for i in range(L, R):
		if comp(value, array[i]):
			array[i], array[pointer] = array[pointer], array[i]
			pointer += 1
	array[pointer], array[R] = value, array[pointer]

	quick_sort_recursive(array, L, pointer - 1)
	quick_sort_recursive(array, pointer + 1, R)


def quick_sort(array: Sequence[Item]) -> NoReturn:
	quick_sort_recursive(array, 0, len(array) - 1)

test_sorter(merge_sort)

#-------------------DIJKSTRA'S ALGORITHM--------------------
def dijkstra_distance(
	start: Key, 
	graph: dict[Key, dict[Key, int]]) -> dict[Key, int]:
	distances = {
		vertex: float('inf') if vertex != start else 0 
		for vertex in graph.keys()
	}
	visited = []

	for _ in range(len(graph)):
		current_vertex = None
		for vertex in distances.keys():
			if vertex not in visited and (current_vertex is None or distances[vertex] < distances[current_vertex]):
				current_vertex = vertex

		for vertex in graph[current_vertex].keys():
			if (new_distance := distances[current_vertex] + graph[current_vertex][vertex]) < distances[vertex]:
				distances[vertex] = new_distance
		visited.append(current_vertex)
	return distances


graph = {
	'A': {'B': 5, 'F': 6},
	'B': {'A': 5, 'F': 1, 'C': 2},
	'C': {'B': 2, 'F': 3, 'E': 1, 'D': 1},
	'F': {'A': 6, 'B': 1, 'C': 3, 'D': 0, 'E': 4},
	'D': {'C': 1, 'E': 1, 'F': 0},
	'E': {'B': 10, 'F': 4, 'C': 1, 'D': 1}
}

dijkstra_distance('A', graph)

#-------------------LINKED LIST--------------------
class LinkedListIndexError(IndexError):
	"""LinkedList is out of range"""

class LinkedList:
	__head = None
	__length = 0

	class Node:
		def __init__(self, element: Item, 
			next_node: Optional['LinkedList.Node'] = None):
			self.element = element
			self.next_node = next_node

		def __str__(self) -> str:
			return str(self.element)

	def __len__(self) -> int:
		return self.__length

	def __nonzero__(self) -> bool:
		return self.__length != 0

	def __str__(self) -> str:
		result = ''
		node = self.__head

		for i in range(len(self)):
			result += str(node)
			if i == len(self) - 1: break
			else: result += ', '
			node = node.next_node

		return f'LinkedList: [{result}]'

	def __reversed__(self) -> 'LinkedList':
		def _reverse_recursive(
			prev_node: 'LinkedList.Node',
			cur_node: 'LinkedList.Node') -> 'LinkedList.Node':
			if cur_node is None: return prev_node
			nxt = cur_node.next_node
			cur_node.next_node = prev_node
			return _reverse_recursive(cur_node, nxt)

		self.__head = _reverse_recursive(None, self.__head)
		return self

	def __getitem__(self, index: int) -> 'LinkedList.Node':
		if not self or index + 1 > len(self):
			raise LinkedListIndexError
			
		i = 0
		node = self.__head
		while i < index:
			node = node.next_node
			i += 1
		return node

	def __setitem__(self, index: int, value: Item) -> NoReturn:
		if not self and not index:
			self.__head = self.Node(value)
			self.__length += 1
			return
		node = self[index - 1]
		node.next_node = self.Node(value, node.next_node.next_node)


	def append(self, value: Item) -> NoReturn:
		self[self.__length - 1].next_node = self.Node(value)
		self.__length += 1
		

	def insert(self, value: Item, index: int) -> NoReturn:
		if index + 1 > len(self) and self:
			raise LinkedListIndexError
		elif not index and not self:
			self[index] = value
			return
		node = self[index]
		self[index] = self.Node(value, node)

	def pop(self, index: int) -> NoReturn:
		if not index:
			self.__head = self[1]
		else:
			node = self[index - 1]
			node.next_node = node.next_node.next_node
		self.__length -= 1


lst = LinkedList()
lst.insert(1000, 0)
lst.append(10)
lst.append(200)
lst.append(4)
lst.insert(30, 2)
lst.append(80)

lst # LinkedList: [1000, 10, 30, 4, 80]
lst.pop(0)
lst[2] = 3000
reversed(lst) #LinkedList: [80, 3000, 30, 10]
