class LabelEncoder:
    classes_: t.Sequence[t.Any]
    def fit(self, x): ...
    def transform(self, x): ...
    def fit_transform(self, x): ...