import json

import six


class Example(object):
    """Defines a single training or test example.

    Stores each column of the example as an attribute.
    """

    @classmethod
    def fromJSON(cls, data: str, fields: dict):
        """
        args:
          data: json str
          fields: dict: old_name -> (new_name, Field)
        """
        return cls.fromdict(json.loads(data), fields)

    @classmethod
    def fromdict(cls, data: dict, fields: dict):
        """
        args:
          data:
          fields: dict: old_name -> (new_name, Field)
        """
        ex = cls()
        for key, vals in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                                 "the input data".format(key))
            if vals is not None:
                if not isinstance(vals, list):
                    vals = [vals]
                for val in vals:
                    name, field = val
                    setattr(ex, name, field.preprocess(data[key]))
        return ex

    @classmethod
    def fromCSV(cls, data, fields, field_to_index=None):
        """
        args:
          data:
          fields:
             list[(str, Field)] if field_to_index is None;
             Dict: old_name -> (new_name, Field) if field_to_index is Not None
          fields_to_index:
        """
        if field_to_index is None:
            return cls.fromlist(data, fields)
        else:
            assert(isinstance(fields, dict))
            data_dict = {f: data[idx] for f, idx in field_to_index.items()}
            return cls.fromdict(data_dict, fields)

    @classmethod
    def fromlist(cls, data: list, fields: list):
        """
        args:
          data: List, 每个元素为对应某个Field的内容, 顺序与fields顺序一致
          fields: List: (str, Field) or ((str, str, ...), (Field, Field, ...)), 一个val对应对个field?
        """
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                if isinstance(val, six.string_types):
                    val = val.rstrip('\n')
                # Handle field tuples ??
                if isinstance(name, tuple):
                    for n, f in zip(name, field):
                        setattr(ex, n, f.preprocess(val))
                else:
                    setattr(ex, name, field.preprocess(val))
        return ex

    @classmethod
    def fromtree(cls, data, fields, subtrees=False):
        try:
            from nltk.tree import Tree
        except ImportError:
            print("Please install NLTK. "
                  "See the docs at http://nltk.org for more information.")
            raise
        tree = Tree.fromstring(data)
        if subtrees:
            return [cls.fromlist(
                [' '.join(t.leaves()), t.label()], fields) for t in tree.subtrees()]
        return cls.fromlist([' '.join(tree.leaves()), tree.label()], fields)
