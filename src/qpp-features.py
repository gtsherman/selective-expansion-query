import argparse
import math
import statistics

import pyndri

from retrieval.core import IndexWrapper, build_rm1, read_queries, Stopper
from retrieval.scoring import clarity, DirichletTermScorer

"""
List of features:
- clarity
- weighted information gain
- normalized query commitment
"""


def main():
    options = argparse.ArgumentParser()
    options.add_argument('queries')
    options.add_argument('stoplist')
    options.add_argument('expansion_indexes', nargs='+')
    args = options.parse_args()

    queries = read_queries(args.queries)
    stopper = Stopper(file=args.stoplist)

    for index_name in args.expansion_indexes:
        index = IndexWrapper(pyndri.Index(index_name))
        for query in queries:
            query.vector = stopper.stop(query.vector)

            if len(query.vector) == 0:
                continue

            top_results = index.query(query, count=10)

            rm1 = build_rm1(top_results, index, stopper=stopper)

            # Features
            rm1_clarity = clarity(rm1.vector, index)
            weighted_ig = wig(query, index, top_results=top_results)
            normalized_qc = nqc(query, index, top_results=top_results)

            print(query.title, index_name, rm1_clarity, weighted_ig, normalized_qc, sep=',')


def wig(query, index, top_results=None):
    if top_results is None:
        top_results = index.query(query, count=10)
    if len(top_results) == 0:
        return 0.0

    dirichlet_scorer = DirichletTermScorer(index)

    wig = 0.0
    for doc, _ in top_results:
        for term in query.vector:
            p_doc = dirichlet_scorer.score(term, doc)
            p_col = (index.term_count(term) + 1) / index.total_terms()
            lam = 1 / math.sqrt(query.length())
            wig += lam * math.log(p_doc / p_col)

    return wig / len(top_results)


def nqc(query, index, top_results=None):
    if top_results is None:
        top_results = index.query(query, count=10)
    if len(top_results) == 0:
        return 0.0

    mu = statistics.mean([score for _, score in top_results])
    squared_diffs = [(score - mu)**2 for _, score in top_results]
    num = math.sqrt(sum(squared_diffs) / len(top_results))

    col_score = 0.0
    for term in query.vector:
        q_weight = query.vector[term] / query.length()
        term_score = math.log((index.term_count(term) / index.total_terms())+0.001)
        col_score += q_weight * term_score

    return num / col_score


if __name__ == '__main__':
    main()
