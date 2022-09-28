import pandas as pd
import plotly.graph_objects as go
from src.Arules import Arules


class AssociationRulesExtractor:
    def __init__(self):
        self.classified_keywords = {}
        self.unused_keywords = self.read_unused_keywords("../raw/unused_keywords.txt")

    @staticmethod
    def read_unused_keywords(file_name: str) -> dict:
        unused_keywords = {}
        all_tags = open(file_name, encoding='utf-8', newline='\n').read().strip().split("\n")
        for pairs in all_tags:
            split = pairs.split(":")
            tag = split[0]
            keywords = set(split[1].split(","))
            unused_keywords[tag] = keywords
        return unused_keywords

    def fit(self, data: pd.DataFrame):
        classified_keywords = {}
        for index in range(len(data)):
            row = data.loc[index]
            row_tags = row["Tag"]
            if type(row_tags) != float:
                for tag in row_tags:
                    if tag not in classified_keywords:
                        classified_keywords[tag] = []
                    classified_keywords.get(tag).append(self.filter_unused_keyword_per_tag(tag, row["Keywords"]))
        self.classified_keywords = classified_keywords

    def filter_unused_keyword_per_tag(self, tag: str, keywords: set) -> set:
        return set(filter(lambda keyword: keyword not in self.unused_keywords.get(tag), keywords))

    def transform(self, args: dict, sort_by: str):
        classified_rules = {}
        classified_item_set = {}
        for tag in self.classified_keywords:
            print(f"{tag}:")
            arg = args.get(tag)
            if arg is None:
                continue
            rules_generator = Arules()
            rules_generator.fit(self.classified_keywords.get(tag))
            frequent_item_sets = rules_generator.get_frequent_item_sets(arg[0], arg[1])
            classified_item_set[tag] = pd.DataFrame(frequent_item_sets)
            print("\t\t\tFrequent item set Done")
            rules_data_frame = rules_generator.get_arules(
                frequent_item_sets=frequent_item_sets,
                min_confidence=arg[2],
                min_lift=arg[3],
                sort_by=sort_by
            )
            print("\t\t\tRules Done")
            print("-----------------------")
            classified_rules[tag] = rules_data_frame
        return classified_item_set, classified_rules

    def fit_transform(self, data: pd.DataFrame, args: dict, sort_by: str):
        self.fit(data)
        return self.transform(args, sort_by)

    @staticmethod
    def write_rules_to_csv(rules: dict):
        for tag in rules:
            data_frame = rules.get(tag)
            data_frame['left'] = data_frame['left'].apply(lambda items: " * ".join(items))
            data_frame['right'] = data_frame['right'].apply(lambda items: " * ".join(items))
            data_frame.to_csv(f"../out/AssociationRules/{tag}_Rules.csv")

    @staticmethod
    def write_frequent_item_set_to_csv(item_sets: dict):
        for tag in item_sets:
            data_frame = item_sets.get(tag)
            data_frame['items'] = data_frame['items'].apply(lambda items: " * ".join(items))
            new_column = [support ** 2 * weight for support, weight in zip(data_frame['support'], data_frame['weight'])]
            data_frame.insert(3, "support*weight", new_column, True)
            data_frame = data_frame.sort_values(by='support*weight', ascending=False)
            data_frame.to_csv(f"../out/FrequentItemSet/{tag}_ItemSet.csv")

    @staticmethod
    def draw_scatter_plot_for_rules(rules, title):
        def get_text_for_rule(item):
            return ('Left: {left}<br>' +
                    'Right: {right}<br>' +
                    'Left Support: {left_support}<br>' +
                    'Right Support: {right_support}<br>' +
                    'Support: {support}<br>' +
                    'Confidence: {confidence}<br>' +
                    'Lift: {lift}<br>').format(left=', '.join(item.left),
                                               right=', '.join(item.right),
                                               left_support=item.left_support,
                                               right_support=item.right_support,
                                               support=item.support,
                                               confidence=item.confidence,
                                               lift=item.lift)

        hover_text = [get_text_for_rule(rule) for rule in rules.itertuples()]
        size_ref = 2. * max(rules['lift']) / (10 ** 2)
        fig = go.Figure(
            data=[go.Scatter(x=rules['support'], y=rules['confidence'], text=hover_text, mode='markers', marker=dict(
                color=rules['lift'],
                size=rules['lift'],
                showscale=True,
                sizeref=size_ref
            ))])

        fig.update_layout(
            title=title,
            xaxis=dict(
                title='Support',
                gridcolor='white',
                type='log',
                gridwidth=2,
            ),
            yaxis=dict(
                title='Confidence',
                gridcolor='white',
                gridwidth=2,
            )
        )
        fig.show()
