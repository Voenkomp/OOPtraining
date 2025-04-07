class MyTreeClf:
    def __init__(
        self, max_depth: int = 5, min_sample_split: int = 2, max_leafs: int = 20
    ):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.max_leafs = max_leafs

    def __repr__(self):
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_sample_split}, max_leafs={self.max_leafs}"


obj1 = MyTreeClf()

print(obj1)
