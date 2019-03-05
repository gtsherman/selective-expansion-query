import argparse
import collections
import math
import statistics

import pyndri

from retrieval.core import Document, IndexWrapper, build_rm1, read_queries, Stopper
from retrieval.scoring import clarity, DirichletTermScorer

"""
List of features:
- clarity
- weighted information gain
- normalized query commitment
"""


def main():
    options = argparse.ArgumentParser()
    options.add_argument('run')
    options.add_argument('index')
    options.add_argument('queries')
    options.add_argument('stoplist')
    args = options.parse_args()

    index = IndexWrapper(pyndri.Index(args.index))
    queries = read_queries(args.queries)
    stopper = Stopper(file=args.stoplist)

    run = collections.defaultdict(list)
    with open(args.run) as f:
        for line in f:
            query, _, doc, _, score, _ = line.strip().split()
            run[query].append((Document(index, docno=doc), float(score)))

    for query in run:
        top_results = run[query][:10]
        query = [q for q in queries if q.title == query][0]

        rm1 = build_rm1(top_results, index, stopper=stopper)

        # Features
        rm1_clarity = clarity(rm1.vector, index)
        weighted_ig = wig(query, index, top_results=top_results)
        normalized_qc = nqc(query, index, top_results=top_results)

        print(query.title, rm1_clarity, weighted_ig, normalized_qc, sep=',')


def wig(query, index, top_results=None):
    if top_results is None:
        top_results = index.query(query, count=10)

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
