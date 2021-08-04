# -*- coding: utf-8 -*-

import attr

from attr.validators import instance_of, optional
from enum import Enum, EnumMeta


def lazy_property(fn):
    attr_name = '_lazy_' + fn.__name__

    @property
    def lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return lazy_property


class StringComparisonMixin(object):
    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return super().__eq__(other)


class POS(StringComparisonMixin, Enum):
    ADJ = 1
    ADP = 2
    ADV = 3
    AUX = 4
    CCONJ = 5
    DET = 6
    INTJ = 7
    NOUN = 8
    NUM = 9
    PART = 10
    PRON = 11
    PROPN = 12
    PUNCT = 13
    SCONJ = 14
    SYM = 15
    VERB = 16
    X = 17

    @property
    def category_name(self):
        return 'POS'

    def help(self):
        return f'https://universaldependencies.org/u/pos/{self.name}.html'

    def __str__(self):
        return self.name

    @classmethod
    def build(cls, value):
        return cls[value]


class GrammarCategory(StringComparisonMixin, Enum):
    @property
    def category_name(self):
        return type(self).__name__

    def help(self):
        return 'https://universaldependencies.org/u/feat/{category}.html#{value}'.format(
            category=self.category_name,
            value=self.name
        )

    def __str__(self):
        return f'{self.category_name}={self.name}'

    @classmethod
    def build(cls, value):
        if value is None:
            return None
        return cls[value]


class Abbr(GrammarCategory):
    Yes = 0


class Animacy(GrammarCategory):
    Anim = 0
    Inan = 1


class Aspect(GrammarCategory):
    Imp = 0
    Perf = 1


class Case(GrammarCategory):
    Acc = 0
    Dat = 1
    Gen = 2
    Ins = 3
    Loc = 4
    Nom = 5
    Par = 6
    Voc = 7


class Degree(GrammarCategory):
    Cmp = 0
    Pos = 1
    Sup = 2


class Foreign(GrammarCategory):
    Yes = 0


class Gender(GrammarCategory):
    Fem = 0
    Masc = 1
    Neut = 2


class Hyph(GrammarCategory):
    Yes = 0


class Mood(GrammarCategory):
    Cnd = 0
    Imp = 1
    Ind = 2


class NumForm(GrammarCategory):
    Digit = 0


class NumType(GrammarCategory):
    Card = 0


class Number(GrammarCategory):
    Plur = 0
    Sing = 1


class PersonEnumMeta(EnumMeta):
    _fix_mapping = {
        '1': 'First',
        '2': 'Second',
        '3': 'Third',
    }

    def __getitem__(self, name):
        return super().__getitem__(self._fix_mapping.get(name, name))


class Person(GrammarCategory, metaclass=PersonEnumMeta):
    First = 1
    Second = 2
    Third = 3


class Polarity(GrammarCategory):
    Neg = 0


class Reflex(GrammarCategory):
    Yes = 0


class Tense(GrammarCategory):
    Fut = 0
    Past = 1
    Pres = 2


class Typo(GrammarCategory):
    Yes = 0


class Variant(GrammarCategory):
    Short = 0


class VerbForm(GrammarCategory):
    Conv = 0
    Fin = 1
    Inf = 2
    Part = 3


class Voice(GrammarCategory):
    Act = 0
    Mid = 1
    Pass = 2


def _to_snake_case(line):
    result = ''
    for letter in line:
        if letter.isupper() and result:
            result += '_'
        result += letter.lower()
    return result


@attr.s
class GrammarValue(object):
    pos = attr.ib(converter=POS.build, validator=instance_of(POS))
    abbr = attr.ib(default=None, converter=Abbr.build, validator=optional(instance_of(Abbr)))
    animacy = attr.ib(default=None, converter=Animacy.build, validator=optional(instance_of(Animacy)))
    aspect = attr.ib(default=None, converter=Aspect.build, validator=optional(instance_of(Aspect)))
    case = attr.ib(default=None, converter=Case.build, validator=optional(instance_of(Case)))
    degree = attr.ib(default=None, converter=Degree.build, validator=optional(instance_of(Degree)))
    foreign = attr.ib(default=None, converter=Foreign.build, validator=optional(instance_of(Foreign)))
    gender = attr.ib(default=None, converter=Gender.build, validator=optional(instance_of(Gender)))
    hyph = attr.ib(default=None, converter=Hyph.build, validator=optional(instance_of(Hyph)))
    mood = attr.ib(default=None, converter=Mood.build, validator=optional(instance_of(Mood)))
    num_form = attr.ib(default=None, converter=NumForm.build, validator=optional(instance_of(NumForm)))
    num_type = attr.ib(default=None, converter=NumType.build, validator=optional(instance_of(NumType)))
    number = attr.ib(default=None, converter=Number.build, validator=optional(instance_of(Number)))
    person = attr.ib(default=None, converter=Person.build, validator=optional(instance_of(Person)))
    polarity = attr.ib(default=None, converter=Polarity.build, validator=optional(instance_of(Polarity)))
    reflex = attr.ib(default=None, converter=Reflex.build, validator=optional(instance_of(Reflex)))
    tense = attr.ib(default=None, converter=Tense.build, validator=optional(instance_of(Tense)))
    typo = attr.ib(default=None, converter=Typo.build, validator=optional(instance_of(Typo)))
    variant = attr.ib(default=None, converter=Variant.build, validator=optional(instance_of(Variant)))
    verb_form = attr.ib(default=None, converter=VerbForm.build, validator=optional(instance_of(VerbForm)))
    voice = attr.ib(default=None, converter=Voice.build, validator=optional(instance_of(Voice)))

    @classmethod
    def from_string(cls, string):
        pos, feats = string.split('|', 1)
        grammar_value = {'pos': pos}

        if feats != '_':
            for feat in feats.split('|'):
                category, value = feat.split('=')
                grammar_value[_to_snake_case(category)] = value

        return cls(**grammar_value)

    @lazy_property
    def string_representation(self):
        fields = attr.asdict(self)
        string_fields = [str(fields['pos'])]
        for field_name, field_val in sorted(fields.items(), key=lambda pair: pair[0]):
            if field_name == 'pos' or field_val is None:
                continue
            string_fields.append(str(field_val))
        if len(string_fields) == 1:
            string_fields.append('_')
        return '|'.join(string_fields)

    def __str__(self):
        return self.string_representation


class SyntaxRelation(Enum):
    punct = 0
    case = 1
    nmod = 2
    amod = 3
    obl = 4
    nsubj = 5
    advmod = 6
    root = 7
    conj = 8
    cc = 9
    obj = 10
    det = 11
    parataxis = 12
    mark = 13
    xcomp = 14
    acl = 15
    appos = 16
    fixed = 17
    advcl = 18
    iobj = 19
    nsubj_pass = 20
    nummod = 21
    acl_relcl = 22
    flat_name = 23
    ccomp = 24
    csubj = 25
    nummod_gov = 26
    cop = 27
    aux_pass = 28
    aux = 29
    flat_foreign = 30
    discourse = 31
    orphan = 32
    flat = 33
    compound = 34
    expl = 35
    obl_agent = 36
    list = 37
    vocative = 38
    nummod_entity = 39
    csubj_pass = 40
    goeswith = 41
    dep = 42
    dislocated = 43
    reparandum = 44

    @classmethod
    def from_string(cls, string):
        return cls[string.replace(':', '_')]

    def help(self):
        return f'https://universaldependencies.org/u/dep/{self.name.replace("_", "-")}.html'

    def __str__(self):
        return self.name.replace('_', ':')
