import argparse
import collections
from statistics import mean

import pyndri

from retrieval.core import read_queries, IndexWrapper, ExpandableDocument, Stopper, build_rm1
from retrieval.scoring import jaccard_similarity, cosine_similarity, clarity


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
        for j in range(i+1, len(docs)):
            sims.append(fun(docs[i], docs[j]))
    return sims


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
        results = target_index.query(query, count=10)
        target_rm = build_rm1(query, target_index, num_docs=10, stopper=stopper)

        # Get the query results in each expansion index
        expansion_result_sets = []
        expansion_rms = []
        for expansion_index in expansion_indexes:
            expansion_results = expansion_index.query(query, count=10)
            expansion_result_sets.append(set([r[0].docno for r in expansion_results]))
            expansion_rms.append(build_rm1(query, expansion_index, num_docs=10, stopper=stopper))

        jaccard_similarities = [[] for _ in expansion_indexes]
        expansion_rm_similarities = [[] for _ in expansion_indexes]
        target_rm_similarities = [[] for _ in expansion_indexes]
        doc_to_expansion_similarities = [[] for _ in expansion_indexes]
        avg_expansion_doc_pairwise_similarities = [[] for _ in expansion_indexes]
        doc_expansion_collection_clarity = [[] for _ in expansion_indexes]
        avg_expansion_doc_clarity = [[] for _ in expansion_indexes]

        for doc, score in results:  # for each top doc,
            for i, expansion_index in enumerate(expansion_indexes):  # one entry per collection
                # Convert to expandable document
                expandable_doc = ExpandableDocument(doc.docno, doc.index, expansion_index=expansion_index)

                # Get expansion docs
                expansion_docs = expandable_doc.expansion_docs(expandable_doc.pseudo_query(stopper=stopper))
                expansion_pseudo_doc = combine_vectors(*[r[0].document_vector() for r in expansion_docs])

                # Compute stuff
                jaccard_similarities[i].append(jaccard_similarity(expansion_result_sets[i], set([r[0].docno for r in
                                                                                                 expansion_docs])))
                expansion_rm_similarities[i].append(cosine_similarity(expansion_rms[i].vector, expansion_pseudo_doc))
                target_rm_similarities[i].append(cosine_similarity(target_rm.vector, expansion_pseudo_doc))
                doc_to_expansion_similarities[i].append(cosine_similarity(doc.document_vector(), expansion_pseudo_doc))
                avg_expansion_doc_pairwise_similarities[i].append(mean(pairwise_similarity(*[r[0].document_vector()
                                                                                             for r in
                                                                                             expansion_docs])))
                doc_expansion_collection_clarity[i].append(clarity(expandable_doc.pseudo_query(stopper=stopper).vector,
                                                                   expansion_index))
                avg_expansion_doc_clarity[i].append(mean([clarity(r[0].document_vector(), expansion_index) for r in
                                                          expansion_docs]))

        jaccard_similarities = each_average(jaccard_similarities)
        expansion_rm_similarities = each_average(expansion_rm_similarities)
        target_rm_similarities = each_average(target_rm_similarities)
        doc_to_expansion_similarities = each_average(doc_to_expansion_similarities)
        avg_expansion_doc_pairwise_similarities = each_average(avg_expansion_doc_pairwise_similarities)
        doc_expansion_collection_clarity = each_average(doc_expansion_collection_clarity)
        avg_expansion_doc_clarity = each_average(avg_expansion_doc_clarity)

        for i in range(len(expansion_indexes)):
            print(query.title, args.expansion_indexes[i], jaccard_similarities[i], expansion_rm_similarities[i],
                  target_rm_similarities[i], doc_to_expansion_similarities[i],
                  avg_expansion_doc_pairwise_similarities[i], doc_expansion_collection_clarity[i],
                  avg_expansion_doc_clarity[i])


if __name__ == '__main__':
    main()
