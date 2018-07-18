import unittest
import random
from Tensor import *

indices = [Index(random.randint(1, 10)) for _ in range(6)]
i, j, k, l, m, n = indices


class Groups(unittest.TestCase):

    def test_flat_commutative_op(self):
        """ops with group axioms should automatically flatten"""
        def is_flat(branch):
            print(branch)
            for child in branch.children:
                # print("TAG")
                # print(child)
                self.assertTrue(issubclass(type(child), Tensor))

        A = Tensor(i, j, k)
        B = Tensor(j, k, l)
        C = Tensor(l, m)

        is_flat(A @ B @ C)
        is_flat(A @ Product(A, B))
        is_flat(Product(A, Product(B, C)))
        is_flat(Product(A, B @ C))


class Products(unittest.TestCase):
    def test_contract_1(self):
        """Contraction over 1 index"""
        graph = Tensor(i, j) @ Tensor(j, k)
        self.assertEqual(set(graph.indices), {i, k})

    def test_contract_2(self):
        """Contraction over 2 indices"""
        graph = Tensor(i, j, k) @ Tensor(j, k, l)
        self.assertEqual(set(graph.indices), {i, l})

    def test_no_contract(self):
        """Matrices which don't conform form higher dimensional tensors"""
        graph = Tensor(i, j, k) @ Tensor(l, m, n)
        self.assertEqual(set(graph.indices), {i, j, k, l, m, n})


class Simplification(unittest.TestCase):
    def test_identity(self):
        """Relabel index in simplification"""
        graph = simplify(Tensor(i, j, k) @ Identity(k, l))
        self.assertEqual(set(graph.indices), {i, j, l})

    def test_singleton_binary_op(self):
        """A non-unary op with one child is equivalent to child"""
        graph = simplify(Product(Tensor(i)))
        self.assertTrue(issubclass(type(graph), Tensor))


if __name__ == '__main__':
    unittest.main()
