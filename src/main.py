from src.AssociationRulesExtracter import AssociationRulesExtractor
from src.Utils import read_cleaned_excel


def main():
    # Reading dataset
    data = read_cleaned_excel("../raw/TaggedDataset.xlsx")
    # Building an object from AssociationRulesExtractor
    rules_generator = AssociationRulesExtractor()
    # Extracting association rules and frequent item sets
    item_sets, rules = rules_generator.fit_transform(data, get_association_rules_args(), 'support')
    # Storing frequent item sets
    rules_generator.write_frequent_item_set_to_csv(item_sets)
    # Storing association rules
    rules_generator.write_rules_to_csv(rules)
    # Drawing scatter plot for rules per each tag
    for rule in rules:
        rules_generator.draw_scatter_plot_for_rules(rules.get(rule), rule)


def get_association_rules_args():
    return {
        # min_support, min_weight, min_confidence, min_leaf
        'FRONT_END': (0.10, 0.1, 0.1, 1),
        'BACK_END': (0.1, 0.1, 0.1, 1),
        'MOBILE': (0.06, 0.1, 0.1, 1),
        'AI': (0.06, 0.1, 0.1, 1),
        'MANAGING': (0.05, 0.1, 0.1, 1),
        'NETWORK': (0.05, 0.1, 0.1, 1),
        'DEVOPS': (0.1, 0.1, 0.1, 1),
        'DATABASE': (0.04, 0.1, 0.1, 1),
        'FULL_STACK': (0.07, 0.1, 0.1, 1),
        'SECURITY': (0.05, 0.1, 0.1, 1),
        'SYS_ADMIN': (0.06, 0.1, 0.1, 1),
        'OS': (0.05, 0.1, 0.1, 1),
        'TEST': (0.05, 0.1, 0.1, 1)
    }


if __name__ == '__main__':
    main()
