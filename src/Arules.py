from collections import namedtuple
from itertools import chain
from itertools import combinations
import pandas as pd


class Arules:
    def __init__(self):
        self.data = []
        self.transactions_map = {}
        self.items = set()
        self.transaction_count = 0

    def fit(self, data: list):
        self.data = data
        self.transactions_map, self.items, self.transaction_count = self.convert_data(data)

    """
        get frequent patterns
    """

    def get_frequent_item_sets(self, min_support, min_weight):
        # used in k-item expression
        next_k = 1
        # C(K)
        candidates = [frozenset([item]) for item in self.items]
        # loop until k reached to max
        while candidates:
            # L(K)
            larges = set()
            # convert C(K) to L(K)
            for candidate in candidates:
                support, weight = self.calculate_support(candidate)
                if support**2 * weight < min_support**2 * min_weight:
                    continue
                larges.add(candidate)
                # generate an iterator which iterate throughout frequent item sets
                frequent_item_set = FrequentItemSet(candidate, support, weight)
                print(f"\tfrequent item set:{frequent_item_set}")
                yield frequent_item_set
            next_k += 1
            # generate next k-item candidate from current large
            candidates = self.generate_next_candidates(larges, next_k)

    """
        get association rules
    """

    def get_arules(self, frequent_item_sets, min_confidence=None, min_lift=None, sort_by='lift'):
        # generate association rules
        rules = list(chain.from_iterable(self.generate_and_filter_rules(frequent_item_sets, min_confidence, min_lift)))
        rules = pd.DataFrame(rules, columns=['left',
                                             'right',
                                             'left_support',
                                             'left_weight',
                                             'right_support',
                                             'right_weight',
                                             'support',
                                             'weight',
                                             'confidence',
                                             'lift'])

        return rules.sort_values(by=sort_by, ascending=False)

    """
        generate all rules from frequent item sets and ignore rules which their confidence is lower than min_confidence 
        and their lift is lower than min_lift
    """

    def generate_and_filter_rules(self, frequent_item_sets, min_confidence, min_lift):
        for frequent_item_set in frequent_item_sets:
            # ignore 1-item set
            if len(frequent_item_set.items) < 2:
                continue
            rules = [rule for rule in self.generate_rules_from_single_frequent_item_set(frequent_item_set)
                     if rule.confidence >= min_confidence and rule.lift >= min_lift]
            # ignore inappropriate rules
            if not rules:
                continue
            yield rules

    """
        extract all possible rules from every frequent item set
    """

    def generate_rules_from_single_frequent_item_set(self, frequent_item_set):
        items = frequent_item_set.items
        # sort frequent item set by alphabet
        sorted_items = sorted(items)

        # divide item set into to section, left -> right
        # begin from one: item1 -> item2, item3, ... , item(n-1)
        # ends in len(items) -> item1, item2, ... , item(n-1) -> item(n)
        for cursor in range(1, len(items)):
            # generate every combination of frequent item set
            for combination_set in combinations(sorted_items, cursor):
                # divide into to section left -> right
                left = frozenset(combination_set)
                right = frozenset(items.difference(left))
                left_support, left_weight = self.calculate_support(left)
                right_support, right_weight = self.calculate_support(right)
                # confidence = support(left U right) / support(left)
                confidence = (frequent_item_set.support / left_support)
                # lift = confidence / support(right)
                lift = confidence / right_support
                rule = AssociationRule(left,
                                       right,
                                       left_support,
                                       left_weight,
                                       right_support,
                                       right_weight,
                                       frequent_item_set.support,
                                       frequent_item_set.weight,
                                       confidence,
                                       lift)
                print(f"\t\trule: {rule}")
                yield rule

    """
        calculate support for given items
    """

    def calculate_support(self, items):
        # if items is empty the support is 1
        if not items:
            return 1.0

        # if there is no transaction, support is 0
        if not self.transaction_count:
            return 0.0

        all_indexes = None
        for item in items:
            indexes = self.transactions_map.get(item)
            # there is no transaction which contains this item, so support is 0
            if indexes is None:
                return 0.0
            # union all indexes
            if all_indexes is None:
                all_indexes = indexes
            else:
                all_indexes = all_indexes.intersection(indexes)

        support = float(len(all_indexes)) / self.transaction_count
        count = 0
        for index in all_indexes:
            count += len(self.data[index])
        if count == 0:
            weight = 0
        else:
            weight = len(all_indexes) * len(items) / count
        return support, weight

    """
        generate next candidate base on previous large set
    """

    @staticmethod
    def generate_next_candidates(prev_large, current_k):
        # convert list of frozen set of items to one frozen set and sort by alphabet, in other words next line will
        # extract list of items
        items = sorted(frozenset(chain.from_iterable(prev_large)))

        # generate next candidate (k-item set iterator) from prev_large. note that this candidate is not next candidate,
        # because any subset of candidates must be frequent, so we must evaluate that any subset of this brand new
        # candidate is exist in prev_large or not.
        maybe_next_candidates = (frozenset(x) for x in combinations(items, current_k))

        # if current_k is 2, the subsets of next candidates are the same as 1-item, so they are exist in prev_large and
        # they are surely frequent.
        if current_k < 3:
            return list(maybe_next_candidates)

        # evaluate that every subset of brand new candidate is frequent. in other words, every subset of brand new
        # candidate must exist in prev_large, because every items in prev_large are frequent.
        next_candidates = [
            candidate for candidate in maybe_next_candidates
            if all(frozenset(x) in prev_large for x in combinations(candidate, current_k - 1))
        ]
        return next_candidates

    """
        read a csv file and convert it to desired data structure
    """

    @staticmethod
    def convert_data(transactions: list):
        n_row = len(transactions)
        m_map = {}
        items = set()
        for row in range(n_row):
            for item in transactions[row]:
                if item not in m_map:
                    m_map[item] = set()
                    items.add(item)
                m_map[item].add(row)
        return m_map, items, n_row


FrequentItemSet = namedtuple('FrequentItemSet', ('items', 'support', 'weight'))
AssociationRule = namedtuple('AssociationRule',
                             ('left', 'right',
                              'left_support', 'left_weight',
                              'right_support', 'right_weight',
                              'support', 'weight',
                              'confidence',
                              'lift'))
