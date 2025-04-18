# app.py - Main Flask Application

from flask import Flask, render_template, request, jsonify
import nltk
import numpy as np
import pandas as pd
import re
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datasets import load_dataset

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class LegalIRS:
    def __init__(self):
        self.documents = []
        self.doc_titles = []
        self.inverted_index = {}
        self.doc_vectors = []
        self.idf_values = {}
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.load_data()
        self.preprocess()
        self.create_inverted_index()
        self.calculate_tf_idf()
    
    def load_data(self):
        """Create a dummy dataset of legal documents"""
        print("Creating dummy legal documents dataset...")
        
        # Sample legal topics and keywords to generate document titles and content
        legal_topics = [
            "Constitutional Law", "Criminal Law", "Civil Law", "Property Law", 
            "Contract Law", "Family Law", "Environmental Law", "Tax Law",
            "Labor Law", "Corporate Law", "Intellectual Property", "Human Rights",
            "Administrative Law", "Banking Law", "Consumer Protection"
        ]
        
        legal_keywords = [
            "rights", "justice", "petition", "judicial review", "amendment", "statute",
            "provision", "penalty", "liability", "prosecution", "plaintiff", "defendant",
            "evidence", "testimony", "verdict", "judgment", "appeal", "precedent",
            "jurisdiction", "legislation", "regulation", "compliance", "violation",
            "compensation", "damages", "contract", "agreement", "property", "ownership",
            "dispute", "resolution", "arbitration", "mediation", "enforcement"
        ]
        
        # Generate 200 dummy documents
        import random
        
        for i in range(200):
            # Generate title
            topic = random.choice(legal_topics)
            act_number = random.randint(1950, 2023)
            title = f"{topic} Act, {act_number}"
            
            # Generate content
            content_parts = []
            num_sections = random.randint(3, 10)
            
            # Introduction
            content_parts.append(f"THE {topic.upper()} ACT, {act_number}\n")
            content_parts.append(f"An Act to provide for {topic.lower()} and matters connected therewith.\n")
            content_parts.append(f"BE it enacted by Parliament in the {act_number} Year of the Republic as follows:-\n\n")
            
            # Sections
            for section in range(1, num_sections + 1):
                section_title = f"Section {section}: {random.choice(['Definitions', 'Scope', 'Application', 'Powers', 'Duties', 'Restrictions', 'Exemptions', 'Procedures', 'Penalties', 'Miscellaneous'])}"
                content_parts.append(f"{section_title}\n")
                
                # Add paragraphs to each section
                num_paras = random.randint(1, 3)
                for p in range(num_paras):
                    # Generate a paragraph with legal terms
                    words = random.sample(legal_keywords, min(len(legal_keywords), random.randint(5, 15)))
                    paragraph = " ".join([
                        "The",
                        random.choice(["court", "authority", "government", "state", "petitioner", "respondent", "plaintiff", "defendant"]),
                        random.choice(["shall", "may", "must", "can", "will"]),
                        random.choice(["consider", "examine", "review", "analyze", "determine", "decide"]),
                        "the",
                        random.choice(["matter", "case", "issue", "subject", "petition", "application"]),
                        "regarding",
                        " and ".join(words),
                        "."
                    ])
                    content_parts.append(f"{paragraph}\n")
                
                content_parts.append("\n")
            
            # Add document to our corpus
            self.doc_titles.append(title)
            self.documents.append("".join(content_parts))
        
        print(f"Created {len(self.documents)} dummy legal documents")
        
    def preprocess_text(self, text):
        """Preprocess text: lowercase, tokenize, remove stopwords, and stem"""
        # Lowercase and tokenize
        tokens = nltk.word_tokenize(text.lower())
        
        # Remove non-alphanumeric and stopwords
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token.isalnum() and token not in self.stop_words]
        
        return tokens
    
    def preprocess(self):
        """Preprocess all documents in the corpus"""
        print("Preprocessing documents...")
        self.preprocessed_docs = []
        for doc in self.documents:
            self.preprocessed_docs.append(self.preprocess_text(doc))
    
    def create_inverted_index(self):
        """Create an inverted index from the preprocessed documents"""
        print("Creating inverted index...")
        for doc_id, tokens in enumerate(self.preprocessed_docs):
            for token in set(tokens):  # Use set to count each term once per document
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append(doc_id)
    
    def calculate_tf_idf(self):
        """Calculate TF-IDF vectors for all documents"""
        print("Calculating TF-IDF vectors...")
        # Get all unique terms
        all_terms = list(self.inverted_index.keys())
        
        # Calculate IDF for each term
        N = len(self.documents)
        for term in all_terms:
            df = len(self.inverted_index[term])  # Document frequency
            self.idf_values[term] = math.log10(N / df)
        
        # Calculate TF-IDF vector for each document
        for doc_id, tokens in enumerate(self.preprocessed_docs):
            doc_vector = {}
            # Count term frequencies
            term_freq = {}
            for token in tokens:
                if token not in term_freq:
                    term_freq[token] = 0
                term_freq[token] += 1
            
            # Calculate TF-IDF for each term in document
            for term, freq in term_freq.items():
                tf = freq / len(tokens)
                doc_vector[term] = tf * self.idf_values.get(term, 0)
            
            self.doc_vectors.append(doc_vector)
    
    def boolean_query(self, query_text):
        """Process a boolean query (supports AND, OR, NOT)"""
        # Simple parser for boolean queries
        query_text = query_text.lower()
        
        # Extract operators and terms
        if " and " in query_text:
            parts = query_text.split(" and ")
            op = "AND"
        elif " or " in query_text:
            parts = query_text.split(" or ")
            op = "OR"
        else:
            # Single term or NOT term
            if query_text.startswith("not "):
                term = self.preprocess_text(query_text[4:])[0] if len(self.preprocess_text(query_text[4:])) > 0 else ""
                if term and term in self.inverted_index:
                    all_docs = set(range(len(self.documents)))
                    return list(all_docs - set(self.inverted_index[term]))
                return []
            else:
                term = self.preprocess_text(query_text)[0] if len(self.preprocess_text(query_text)) > 0 else ""
                return self.inverted_index.get(term, []) if term else []
        
        # Process each part of the query
        results = []
        for part in parts:
            if part.startswith("not "):
                term = self.preprocess_text(part[4:])[0] if len(self.preprocess_text(part[4:])) > 0 else ""
                if term and term in self.inverted_index:
                    all_docs = set(range(len(self.documents)))
                    part_result = list(all_docs - set(self.inverted_index[term]))
                else:
                    part_result = list(range(len(self.documents)))
            else:
                term = self.preprocess_text(part)[0] if len(self.preprocess_text(part)) > 0 else ""
                part_result = self.inverted_index.get(term, []) if term else []
            
            results.append(set(part_result))
        
        # Combine results based on operator
        if op == "AND":
            final_result = results[0]
            for r in results[1:]:
                final_result = final_result.intersection(r)
        else:  # OR
            final_result = results[0]
            for r in results[1:]:
                final_result = final_result.union(r)
        
        return list(final_result)
    
    def vector_query(self, query_text):
        """Process a query using the vector space model with TF-IDF weighting"""
        # Preprocess query
        query_terms = self.preprocess_text(query_text)
        
        # Create query vector
        query_vector = {}
        # Count term frequencies in query
        for term in query_terms:
            if term not in query_vector:
                query_vector[term] = 0
            query_vector[term] += 1
        
        # Calculate TF-IDF for query terms
        for term, freq in query_vector.items():
            tf = freq / len(query_terms)
            query_vector[term] = tf * self.idf_values.get(term, 0)
        
        # Calculate cosine similarity between query and documents
        scores = []
        for doc_id, doc_vector in enumerate(self.doc_vectors):
            # Compute dot product
            dot_product = 0
            for term, weight in query_vector.items():
                if term in doc_vector:
                    dot_product += weight * doc_vector[term]
            
            # Compute magnitudes
            query_mag = math.sqrt(sum(w**2 for w in query_vector.values()))
            doc_mag = math.sqrt(sum(w**2 for w in doc_vector.values()))
            
            # Compute cosine similarity
            similarity = dot_product / (query_mag * doc_mag) if query_mag > 0 and doc_mag > 0 else 0
            scores.append((doc_id, similarity))
        
        # Sort by similarity score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return document IDs of the top matching documents
        return [(doc_id, score) for doc_id, score in scores if score > 0]

# Initialize the IRS
irs = LegalIRS()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    search_type = request.form['search_type']
    
    results = []
    if search_type == 'boolean':
        doc_ids = irs.boolean_query(query)
        for doc_id in doc_ids:
            results.append({
                'id': doc_id,
                'title': irs.doc_titles[doc_id],
                'snippet': irs.documents[doc_id][:200] + '...' if len(irs.documents[doc_id]) > 200 else irs.documents[doc_id]
            })
    else:  # vector
        doc_ids_with_scores = irs.vector_query(query)
        for doc_id, score in doc_ids_with_scores[:10]:  
            results.append({
                'id': doc_id,
                'title': irs.doc_titles[doc_id],
                'snippet': irs.documents[doc_id][:200] + '...' if len(irs.documents[doc_id]) > 200 else irs.documents[doc_id],
                'score': round(score, 4)
            })
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)