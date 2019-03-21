import argparse
import collections
from functools import reduce
from operator import mul
from statistics import mean

import pyndri

from retrieval.core import read_queries, IndexWrapper, ExpandableDocument, Stopper, build_rm1, Query
from retrieval.scoring import jaccard_similarity, cosine_similarity, clarity, average_precision


def combine_vectors(*vectors):
    vector = collections.Counter()
    for v in vectors:
        vector += v
    return vector


def each_average(list_of_lists, avg_fun=mean):
    return [avg_fun(i) for i in list_of_lists]


def pairwise_similarity(*docs, fun=cosine_similarity):
    sims = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            sims.append(fun(docs[i], docs[j]))
    return sims


def collection_ql(query, index):
    return reduce(mul, [((index.term_count(term) + 1) / index.total_terms()) ** query.vector[term] for term in
                        query.vector], 1)


def interpolate(vector1, vector2, vector1_weight):
    def r(vector, weight):
        v = collections.defaultdict(float)
        for term in vector:
            orig_weight = vector[term]
            v[term] += weight * orig_weight
        return v

    return combine_vectors(r(vector1, vector1_weight), r(vector2, 1.0-vector1_weight))


def main():
    options = argparse.ArgumentParser()
    options.add_argument('queries')
    options.add_argument('index')
    options.add_argument('expansion_indexes', nargs='+')
    options.add_argument('stoplist')
    options.add_argument('-q', '--query', help='The title of a specific query to run.')
    args = options.parse_args()

    target_index = IndexWrapper(pyndri.Index(args.index))
    expansion_indexes = [IndexWrapper(pyndri.Index(index)) for index in args.expansion_indexes]

    queries = read_queries(args.queries, format=args.queries.split('.')[-1])

    if args.query is not None:
        for query in queries:
            if query.title == args.query:
                queries = [query]

    stopper = Stopper(file=args.stoplist)

    for query in queries:
        query.vector = stopper.stop(query.vector)

        if len(query.vector) == 0:
            continue

        results = target_index.query(query, count=10)
        target_rm = build_rm1(target_index.query(query, count=10), target_index, stopper=stopper)

        # Get the query results in each expansion index
        expansion_result_sets = []
        expansion_rms = []
        expansion_qls = []
        for expansion_index in expansion_indexes:
            expansion_results = expansion_index.query(query, count=10)
            expansion_result_sets.append(set([r[0].docno for r in expansion_results]))
            expansion_rms.append(build_rm1(expansion_index.query(query, count=10), expansion_index, stopper=stopper))
            expansion_qls.append(collection_ql(query, expansion_index))

        jaccard_similarities = [[] for _ in expansion_indexes]
        pseudo_average_precisions = [[] for _ in expansion_indexes]
        expansion_rm_similarities = [[] for _ in expansion_indexes]
        target_rm_similarities = [[] for _ in expansion_indexes]
        doc_to_expansion_similarities = [[] for _ in expansion_indexes]
        avg_expansion_doc_pairwise_similarities = [[] for _ in expansion_indexes]
        doc_expansion_collection_clarity = [[] for _ in expansion_indexes]
        avg_expansion_doc_clarity = [[] for _ in expansion_indexes]
        exp_pseudo_doc_clarity = [[] for _ in expansion_indexes]
        robustness = [[] for _ in expansion_indexes]

        expansion_pseudo_docs = [[] for _ in expansion_indexes]
        expansion_docs_sets = [[] for _ in expansion_indexes]

        # Get a list of lists:
        #  - doc: feature_value
        #  - expansion_index: [doc1, doc2, ...]
        #  - feature: [expansion_index1, expansion_index2, ...]
        for doc, score in results:  # for each top doc,
            for i, expansion_index in enumerate(expansion_indexes):  # one entry per collection
                # Convert to expandable document
                expandable_doc = ExpandableDocument(doc.docno, doc.index, expansion_index=expansion_index)

                # Get expansion docs
                pseudo_query = expandable_doc.pseudo_query(stopper=stopper)
                expansion_docs = expandable_doc.expansion_docs(pseudo_query, include_scores=False)
                expansion_pseudo_doc = stopper.stop(combine_vectors(*[r.document_vector() for r in expansion_docs]))
                expansion_docs_set = set([r.docno for r in expansion_docs])

                expansion_docs_with_query = set([r.docno for r in expandable_doc.expansion_docs(
                    Query(doc.docno, vector=interpolate(pseudo_query.vector, query.vector, 0.5)),
                    include_scores=False)])

                # Store stuff
                expansion_pseudo_docs[i].append(expansion_pseudo_doc)
                expansion_docs_sets[i].append(expansion_docs_set)

                # Compute stuff
                jaccard_similarities[i].append(jaccard_similarity(expansion_result_sets[i], expansion_docs_set))
                pseudo_average_precisions[i].append(average_precision([r.docno for r in expansion_docs],
                                                                      expansion_result_sets[i]))
                expansion_rm_similarities[i].append(cosine_similarity(stopper.stop(expansion_rms[i].vector),
                                                                      expansion_pseudo_doc))
                target_rm_similarities[i].append(cosine_similarity(stopper.stop(target_rm.vector),
                                                                   expansion_pseudo_doc))
                doc_to_expansion_similarities[i].append(cosine_similarity(stopper.stop(doc.document_vector()),
                                                                          expansion_pseudo_doc))
                avg_expansion_doc_pairwise_similarities[i].append(mean(pairwise_similarity(*[
                    stopper.stop(r.document_vector()) for r in expansion_docs])))
                doc_expansion_collection_clarity[i].append(clarity(expandable_doc.pseudo_query(stopper=stopper).vector,
                                                                   expansion_index))
                avg_expansion_doc_clarity[i].append(mean([clarity(stopper.stop(r.document_vector()), expansion_index)
                                                                  for r in expansion_docs]))
                exp_pseudo_doc_clarity[i].append(clarity(stopper.stop(expansion_pseudo_doc), expansion_index))
                robustness[i].append(len(expansion_docs_set & expansion_docs_with_query))

        jaccard_similarities = each_average(jaccard_similarities)
        pseudo_average_precisions = each_average(pseudo_average_precisions)
        expansion_rm_similarities = each_average(expansion_rm_similarities)
        target_rm_similarities = each_average(target_rm_similarities)
        doc_to_expansion_similarities = each_average(doc_to_expansion_similarities)
        avg_expansion_doc_pairwise_similarities = each_average(avg_expansion_doc_pairwise_similarities)
        doc_expansion_collection_clarity = each_average(doc_expansion_collection_clarity)
        avg_expansion_doc_clarity = each_average(avg_expansion_doc_clarity)
        exp_pseudo_doc_clarity = each_average(exp_pseudo_doc_clarity)
        robustness = each_average(robustness)

        for i in range(len(expansion_indexes)):
            print(query.title,
                  args.expansion_indexes[i],
                  jaccard_similarities[i],  # jaccard
                  expansion_rm_similarities[i],  # expansion_rm
                  target_rm_similarities[i],  # target_rm
                  doc_to_expansion_similarities[i],  # doc_vs_expansion
                  avg_expansion_doc_pairwise_similarities[i],  # pairwise
                  doc_expansion_collection_clarity[i],  # doc_clarity
                  avg_expansion_doc_clarity[i],  # exp_doc_clarity
                  exp_pseudo_doc_clarity[i],  # exp_pseudo_doc_clarity
                  expansion_qls[i],  # exp_ql
                  pseudo_average_precisions[i],  # pseudo_map
                  mean(pairwise_similarity(*expansion_pseudo_docs[i])),  # pairwise_pseudo
                  mean(pairwise_similarity(*expansion_docs_sets[i], fun=jaccard_similarity)),  # pairwise_jaccard
                  len(set.union(*expansion_docs_sets[i])),  # unique_exp
                  robustness[i]  # robustness
                  )


if __name__ == '__main__':
    main()
