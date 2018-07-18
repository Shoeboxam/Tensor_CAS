import unittest
import random
from Tensor import *

indices = [Index(random.randint(1, 10)) for _ in range(6)]
i, j, k, l, m, n = indices


class Products(unittest.TestCase):
    def test_contract_1(self):
        """Contraction over 1 index"""
        graph = Constant((i, j)) @ Constant((j, k))
        self.assertEqual(len(graph.indices), 2)

    def test_contract_2(self):
        """Contraction over 2 indices"""
        graph = Constant((i, j, k)) @ Constant((j, k, l))
        self.assertEqual(len(graph.indices), 2)

    def test_flat_commutative_op(self):
        """Nested commutative operations are invalid"""
        def is_flat(branch):
            for child in branch.children:
                self.assertTrue(issubclass(type(child), Tensor))

        A = Constant((i, j, k))
        B = Constant((j, k, l))
        C = Constant((l, m))

        is_flat(A @ B @ C)
        is_flat(A @ Product([A, B]))
        is_flat(Product([A, Product((B, C))]))
        is_flat(Product([A, B @ C]))


class Simplification(unittest.TestCase):
    def test_identity(self):
        """Relabel index in simplification"""
        graph = simplify(Constant((i, j, k)) @ Identity((k, l)))
        self.assertEqual(set(graph.indices), {i, j, l})


if __name__ == '__main__':
    unittest.main()