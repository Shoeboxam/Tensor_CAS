
# just for pretty printing
SUP = str.maketrans("0123456789abcdefghijklmnopqrstuvwxyz", "⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖᵟʳˢᵗᵘᵛʷˣʸᶻ")


class Index(object):
    def __init__(self, length):
        self._length = length

    def __len__(self):
        return self._length

    def __repr__(self):
        return str(len(self)) + str.translate(self.id, SUP)

    # Collisions are impossible, since they are handled by object ID
    @property
    def id(self):
        return chr(id(self) % 26 + 97)  # 65 for uppercase

    @staticmethod
    def shape(indices):
        return [len(index) for index in indices]

    def __hash__(self):
        return id(self)


# Element of functional graph
class Node(object):
    def __init__(self, children):
        if type(children) not in [list, tuple]:
            children = [children]
        self.children = children

    @property
    def indices(self):
        return set.union(*[child.indices for child in self.children])

    def __call__(self, stimulus):
        return self.propagate([child(stimulus) for child in self.children])

    def gradient(self, variable, indices=None):
        if not indices:
            indices = [Index(len(ind)) for ind in variable.indices]
        # Holds derivatives for each branch of the function graph at the current node
        branches = []
        for child in self.children:
            if variable in child.variables:
                branches.append(self.backpropagate(variable, indices, child) @ child.gradient(variable, indices))

        return Sum(branches)

    @property
    def variables(self):
        """List the input variables"""
        return [var for child in self.children for var in child.variables]

    def __matmul__(*args):
        return Product(args)

    def propagate(self, features):
        raise NotImplementedError(self.__class__.__name__ + ' has not implemented "propagate"')

    def backpropagate(self, variable, indices, child):
        raise NotImplementedError(self.__class__.__name__ + ' has not implemented "backpropagate"')

    def __str__(self):
        raise NotImplementedError(self.__class__.__name__ + ' has not implemented "__str__"')

    def __repr__(self):
        return self.__class__.__name__ + str(self)


class Sigmoid(Node):
    def __str__(self):
        return self.__class__.__name__ + '(' + str(self.children[0]) + ')'


# a Node operation that satisfies the group axioms of commutativity, associativity, identity, inverse
class Group(Node):
    symbol = ''

    def __init__(self, children):
        """commutative ops should not be nested (which shows preference for order)"""
        flat = []
        [flat.extend(child.children) if type(child) is type(self) else flat.append(child) for child in children]
        super().__init__(flat)

    def __str__(self):
        return '(' + self.symbol.join([str(child) for child in self.children]) + ')'\
               + ''.join([str.translate(idx.id, SUP) for idx in self.indices])


# Tensor product
class Product(Group):
    symbol = '⨯'

    @property
    def indices(self):
        present = {}
        for child in self.children:
            for indice in child.indices:
                present[indice] = True if indice not in present else not present[indice]

        return [indice for indice, state in present.items() if state]

    def backpropagate(self, variable, indices, child):
        return Product([sibling for sibling in self.children if child is not sibling])


class Sum(Group):
    symbol = '+'


class Multiply(Group):
    symbol = '*'


class Tensor(Node):
    def __init__(self, indices):
        super().__init__([])
        self._indices = list(indices)
        # self._value = value(*Index.shape(indices)) if callable(value) else value

    @property
    def indices(self):
        return self._indices

    def __str__(self):
        return '[' + '⨯'.join([str(index) for index in self.indices]) + ']'

    def gradient(self, variable, indices):
        return self.backpropagate(variable, indices, self)

    def backpropagate(self, variable, indices, child):
        if variable is self:
            return Product([Identity((i, j)) for i, j in zip(self.indices, indices)])
        return 0  # TODO shape conformation

    @property
    def variables(self):
        # Tensors may be differentiated, hence they are variables
        return [self]


class Constant(Tensor):
    pass


# kronecker delta, identity for relabeling indices
class Identity(Tensor):
    def __init__(self, indices):
        super().__init__(indices)

    @property
    def indices(self):
        return self._indices

    def propagate(self, features):
        return features

    def __str__(self):
        return 'δ' + ''.join([str.translate(idx.id, SUP) for idx in self.indices])


class Variable(Tensor):
    def __init__(self, indices, tag):
        super().__init__(indices)
        self._tag = tag

    def __call__(self, stimulus):
        return self.propagate(stimulus)

    def propagate(self, stimulus):
        return stimulus[self._tag]


def simplify(graph):
    # Single Tensors do not have simplifications
    if issubclass(type(graph), Tensor):
        return graph

    # remove singleton non-unary ops
    if issubclass(type(graph), Group) and len(graph.children) == 1:
        return simplify(graph.children[0])

    def relabel_index(sibling, top, bottom):
        if issubclass(type(sibling), Tensor):
            sibling._indices[sibling._indices.index(top)] = bottom
            return True
        else:
            for nested_child in sibling.children:
                if top in sibling.indices:
                    relabel_index(nested_child, top, bottom)
            return False

    def apply_kronecker(delta):
        top, bottom = delta.indices
        for sibling in [sibling for sibling in graph.children if child is not sibling]:
            if top in sibling.indices:
                return relabel_index(sibling, top, bottom)

    # apply kronecker delta
    for child in reversed(graph.children):
        if type(child) is Identity:
            if apply_kronecker(child):
                graph.children.remove(child)

    graph.children = [simplify(child) for child in graph.children]
    return graph
