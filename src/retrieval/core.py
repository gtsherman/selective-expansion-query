import collections
import json
import math
import xml.etree.ElementTree

from retrieval.scoring import DirichletTermScorer


def read_queries(file_name, format='json'):
    queries = []

    if format == 'json':
        with open(file_name) as f:
            query_data = json.load(f)

        for qd in query_data['queries']:
            title = qd['title']
            vector = {feature['feature']: int(feature['weight']) for feature in qd['model']}
            queries.append(Query(title, vector=vector))
    elif format == 'title':
        tree = xml.etree.ElementTree.parse(file_name)
        root = tree.getroot()
        for qd in root:
            title = qd.find('number').text
            text = qd.find('text').text
            queries.append(Query(title, query_string=text))
    else:
        raise ValueError('Format must be either "json" or "title"')

    return queries


class Query(object):
    def __init__(self, title, query_string='', vector=None):
        self.title = title
        self.vector = collections.Counter(vector)
        for term in query_string.lower().strip().split():
            self.vector[term] += 1

    def length(self):
        return sum(self.vector.values())

    def __str__(self):
        return '#weight( ' + ' '.join([str(weight) + ' ' + term for term, weight in self.vector.items()]) + ' )'


class Stopper(object):
    def __init__(self, terms=None, file=None):
        self.stopwords = set()
        if terms:
            self.stopwords = set(terms)
        if file:
            with open(file) as f:
                for line in f:
                    self.stopwords.add(line.strip().lower())

    def stop(self, vector):
        """
        Return a copy of the vector without stop words.
        :param vector: Assumes the vector is a {term: weight} dictionary, generally a Counter.
        :return: A Counter object containing the vector less stop words.
        """
        return collections.Counter({term: weight for term, weight in vector.items() if term not in self.stopwords})


class Document(object):
    def __init__(self, index, docno=None, doc_id=None):
        self.index = index

        if docno is None and doc_id is None:
            raise ValueError('Must provide either docno or document ID.')

        if doc_id is None:
            self.docno = docno
            try:
                self.doc_id = self.index.doc_id(self.docno)
            except IOError:
                self.doc_id = -1
        elif docno is None:
            self.doc_id = doc_id
            try:
                self.docno = self.index.docno(self.doc_id)
            except IOError:
                self.docno = ''

    def document_vector(self):
        return self.index.document_vector(self.doc_id)

    def __str__(self):
        return '<{docno}>'.format(docno=self.docno)


class ExpandableDocument(Document):
    def __init__(self, docno, index, expansion_index=None):
        super().__init__(index, docno=docno)
        if expansion_index:
            self.expansion_index = expansion_index
        else:
            self.expansion_index = index

    def expansion_docs(self, pseudo_query, num_docs=10, include_scores=True):
        """
        Get the expansion documents for this document.
        :param pseudo_query: The result of calling pseudo_query() on this document, with desired parameters.
        :param num_docs: The number of expansion documents.
        :param include_scores: If True, return a list of (doc, score) tuples.
        :return: A list of ExpandableDocument objects, with corresponding scores if include_scores=True.
        """
        # Get raw expansion docs
        exp_doc_results = self.expansion_index.query(str(pseudo_query), count=num_docs)

        # Normalize scores
        total_score = sum([score for _, score in exp_doc_results])
        exp_doc_results = [(doc.doc_id, score / total_score) for doc, score in exp_doc_results]

        # Convert doc IDs to ExpandableDocument objects
        expansion_docs = [(ExpandableDocument(self.expansion_index.docno(doc_id), self.expansion_index,
                                              self.expansion_index), score) for doc_id, score in exp_doc_results]

        if include_scores:
            return expansion_docs
        return [result[0] for result in expansion_docs]

    def pseudo_query(self, num_terms=20, stopper=Stopper()):
        """
        Converts this document into a pseudo-query, which is a Query containing the limited representation of the
        document.
        :param num_terms: The number of terms to include in the pseudo-query.
        :param stopper: An optional Stopper object to remove stop words from the pseudo-query. By default,
        an empty Stopper.
        :return: A Query object containing the limited representation of the document.
        """
        return Query(self.docno, vector={term: weight for term, weight in
                                         stopper.stop(self.document_vector()).most_common(num_terms)})


class Qrels(object):
    def __init__(self, file=None):
        self._qrels = collections.defaultdict(set)
        if file:
            with open(file) as f:
                for line in f:
                    query, _, docno, rel = line.strip().split()
                    if int(rel) > 0:
                        self._qrels[query].add(docno)

    def is_rel(self, docno, query_title):
        """
        :param docno:
        :param query:
        :return: True if relevant, False otherwise.
        """
        return docno in self._qrels[query_title]

    def rel_docs(self, query_title):
        return self._qrels[query_title]


class BatchResults(object):
    def __init__(self, file=None):
        self._scores = collections.defaultdict(dict)
        self._docs = collections.defaultdict(list)
        if file:
            with open(file) as f:
                for line in f:
                    query, _, docno, rank, score, run = line.strip().split()
                    self._scores[query][docno] = float(score)
                    self._docs[query].append(docno)

    def query_results(self, query_title):
        return self._docs[query_title]

    def document_query_score(self, docno, query_title):
        return self._scores[query_title][docno]


class IndexWrapper(object):
    def __init__(self, index):
        self.index = index
        self._token2id, self._id2token, self._id2df = self.index.get_dictionary()

    def query(self, query, count=1000):
        """
        :param query: Query object
        :param count: Number of documents to retrieve
        :return: List of Document objects
        """
        results = self.index.query(str(query), results_requested=count)
        docs = []
        for doc_id, score in results:
            doc = Document(self, doc_id=doc_id)
            docs.append((doc, score))
        return docs

    def docno(self, doc_id):
        try:
            return self.index.ext_document_id(doc_id)
        except IndexError:
            raise IndexError('Doc ID {} not found in the index.'.format(str(doc_id)))

    def doc_id(self, docno):
        return self.index.document_ids((docno,))[0][1]

    def document_vector(self, doc_id):
            _, token_ids = self.index.document(doc_id)
            return collections.Counter([self._id2token[token_id] for token_id in token_ids if token_id > 0])

    def term_count(self, term):
        return self.index.term_count(term)

    def total_terms(self):
        return self.index.total_terms()

    def term_document_frequency(self, term):
        try:
            term_id = self._token2id[term]
            return self._id2df[term_id]
        except IOError:
            return 0


def build_rm1(initial_results, index, num_terms=20, stopper=None):
    if stopper is None:
        stopper = Stopper()

    zero_mu_scorer = DirichletTermScorer(index, mu=0)

    term_scores = collections.Counter()

    for doc, score in initial_results:
        for term in doc.document_vector():
            if term not in stopper.stopwords:
                term_scores[term] += zero_mu_scorer.score(term, doc) * math.exp(score) / len(initial_results)

    vector = {term: score for term, score in term_scores.most_common(num_terms)}
    return Query('rm1', vector=vector)
