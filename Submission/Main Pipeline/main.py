import shutil
import csv, sqlite3, urllib.parse, uuid, time, random, re, logging
import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
from newspaper import Article, Config
import requests
import spacy
from fastcoref import spacy_component
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pyabsa import AspectPolarityClassification as APC

VERBOSE_LOGGING = False
 
if not VERBOSE_LOGGING:
    logging.disable(logging.CRITICAL)
else:
    logging.basicConfig(level=logging.INFO)
 
db_path = "project.db"

class GeopoliticalAnalyzer:
    def __init__(self):
        print("📥 Loading AI Models (this may take a minute)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ GPU not available, using CPU (this may be slow)")

        # Initialize Spacy with Coreference
        self.nlp = spacy.load("en_core_web_trf")
        self.nlp.add_pipe("fastcoref")
        # Initialize Translation
        self.trans_model_name = "facebook/nllb-200-distilled-600M"
        self.trans_tokenizer = AutoTokenizer.from_pretrained(self.trans_model_name)
        self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(self.trans_model_name)
        self.trans_model.to(self.device)
        
        model_path = "english"
        # Initialize Sentiment (ABSA)
        self.apc_infer = APC.SentimentClassifier(checkpoint=model_path)

    def translate_to_en(self, text):
        """ Function to translate non-English text to English using the NLLB model. """
        if not text: return ""

        max_length = 512
        tgt_lang_id = self.trans_tokenizer.convert_tokens_to_ids("eng_Latn")
        sentences = [s.strip() for s in text.replace('\n', '. ').split('. ') if s.strip()]
        translated_chunks = []

        for sentence in sentences:
            if not sentence.strip(): continue
            try:
                inputs = self.trans_tokenizer(
                    sentence, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=max_length
                ).to(self.device)
                outputs = self.trans_model.generate(
                    **inputs, 
                    forced_bos_token_id=tgt_lang_id,
                    max_length=max_length
                )
                translated = self.trans_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translated_chunks.extend(translated)
            except Exception as e:
                print(f"Translation error for sentence, skipping: {e}")
                continue
        return " ".join(translated_chunks)

    def extract_actor_sentences(self, text):
        """Function to extract actors (PERSON, ORG, GPE) and the sentences they appear in, with coreference resolution."""
        doc = self.nlp(text, component_cfg={"fastcoref": {"resolve_text": True}})
        resolved_text = doc._.resolved_text
        resolved_doc = self.nlp(resolved_text)


        # Structure: { "Actor Name": {"type": "PERSON", "sentences": []} }
        actor_data = {}

        for sent in resolved_doc.sents:
            for ent in sent.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE"):
                    clean_name = self.normalize_name(ent.text)
                    
                    # If we haven't seen this actor yet, initialize their entry
                    if clean_name not in actor_data:
                        actor_data[clean_name] = {
                            "type": ent.label_,
                            "sentences": []
                        }
                    
                    labeled_sent = sent.text.replace(ent.text, f"[B-ASP]{ent.text}[E-ASP]")
                    actor_data[clean_name]["sentences"].append(labeled_sent)
                    
        return actor_data

    def run_sentiment_inference(self, actor_sentences):
        """Function to run sentiment inference on sentences mentioning each actor"""
        final_results = []
        
        for actor, sentences in actor_sentences.items():
            if not sentences:
                continue
                
            # Run inference on all sentences mentioning this specific actor
            predictions = self.apc_infer.predict(sentences, print_result=False, ignore_error=True)
            
            sentiments = []
            confidences = []
            
            for pred in predictions:
                try:
                    sentiments.append(pred['sentiment'][0])
                    confidences.append(float(pred['confidence'][0]))
                except (IndexError, KeyError):
                    continue
                

            # Majority Vote for Sentiment
            sentiment_counts = Counter(sentiments)
            primary_sentiment = sentiment_counts.most_common(1)[0][0]
            
            # Average Confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Calculate Sentiment Counts
            pos_count = sentiments.count('Positive')
            neg_count = sentiments.count('Negative')
            neu_count = sentiments.count('Neutral')

            final_results.append({
                "actor": actor,
                "sentiment": primary_sentiment,
                "avg_confidence": round(avg_confidence, 4),
                "mention_count": len(sentences),
                "pos_count": pos_count,
                "neg_count": neg_count,
                "neu_count": neu_count,
            })

        return pd.DataFrame(final_results)

    def normalize_name(self, text):
        """Helper function to clean and normalize entity names (e.g., remove "The", fix casing)"""
        name = text.strip().lower()
        if name.startswith("the "):
            name = name[4:]
        return name.title()

def setup_database(db_path, is_rerun=False):
    """Function to set up the SQLite database with necessary tables. If is_rerun is True, it will keep existing tables and data."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Only drop tables if this is NOT a rerun
        if not is_rerun:
            cursor.execute("DROP TABLE IF EXISTS article_entities")
            cursor.execute("DROP TABLE IF EXISTS articles")
            
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS articles (
                article_id TEXT PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                canonical_domain TEXT NOT NULL,
                title TEXT,
                text TEXT,
                publish_date TEXT,
                scrape_timestamp TEXT,
                language TEXT CHECK(length(language) = 2),
                processed INTEGER DEFAULT 0 
            );
            CREATE TABLE IF NOT EXISTS article_entities (
                article_id TEXT NOT NULL,
                entity_text TEXT NOT NULL,
                entity_type TEXT,
                sentiment_score REAL,
                dominant_sentiment TEXT,
                mention_count INTEGER,
                pos_count INTEGER DEFAULT 0,
                neg_count INTEGER DEFAULT 0,
                neu_count INTEGER DEFAULT 0,
                PRIMARY KEY (article_id, entity_text),
                FOREIGN KEY (article_id) REFERENCES articles(article_id) ON DELETE CASCADE
            );
        """)
        try:
            cursor.execute("ALTER TABLE articles ADD COLUMN processed INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass 
        conn.commit()

def insert_article(conn, data):
    """Function to insert an extracted article into the database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO articles (article_id, url, canonical_domain, title, text, publish_date, scrape_timestamp, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (data["article_id"], data["url"], data["canonical_domain"], data["title"], data["text"], 
          data.get("publish_date"), datetime.utcnow().isoformat(), data["language"]))
    conn.commit()
    

def insert_article_entity(conn, data):
    """Function to insert extracted entity sentiment data into the database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO article_entities (
            article_id,
            entity_text,
            entity_type,
            sentiment_score,
            dominant_sentiment,
            mention_count,
            pos_count,
            neg_count,
            neu_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        data["article_id"],
        data["entity_text"],
        data["entity_type"],
        data["sentiment_score"],
        data["dominant_sentiment"],
        data["mention_count"],
        data["pos_count"],
        data["neg_count"],
        data["neu_count"]
    ))
    conn.commit()

def run_aggregations(conn):
    """Function to create unified entity tables that aggregate sentiment data across all articles for each language."""
    cursor = conn.cursor()
    for lang in ['en', 'ru']:
        cursor.execute(f"DROP TABLE IF EXISTS unified_entities_{lang}")
        table_name = f"unified_entities_{lang}"
        cursor.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT ae.entity_text, MAX(ae.entity_type) as entity_type, SUM(ae.mention_count) as total_mentions,
                   SUM(ae.pos_count) as total_pos, SUM(ae.neg_count) as total_neg, SUM(ae.neu_count) as total_neu,
                   AVG(ae.sentiment_score) as avg_confidence, '' as dominant_sentiment
            FROM article_entities ae JOIN articles a ON ae.article_id = a.article_id
            WHERE a.language = '{lang}' GROUP BY ae.entity_text;
        """)
        cursor.execute(f"""
            UPDATE {table_name} SET dominant_sentiment = CASE 
                WHEN total_pos >= total_neg AND total_pos >= total_neu THEN 'Positive'
                WHEN total_neg >= total_pos AND total_neg >= total_neu THEN 'Negative' ELSE 'Neutral' END;
        """)
    conn.commit()

def read_csv(file_path):
    """Function to read URLs from a CSV file. Expects a column named 'URL'."""
    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            return [row['URL'] for row in csv.DictReader(csvfile)]
    except: return []

def scrape_article(url, lang):
    """Function to scrape an article's content while bypassing 403/406 errors by using custom headers and manual HTML fetching."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/' 
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        config = Config()
        config.request_timeout = 20
  
        article = Article(url, language=lang)
        article.download(input_html=response.text)
        article.parse()

        parsed_url = urllib.parse.urlparse(url)
        canonical_domain = parsed_url.netloc
    
        return {
            "article_id": str(uuid.uuid4()),
            "url": url,
            "canonical_domain": canonical_domain,
            "title": article.title,
            "text": article.text,
            "publish_date": article.publish_date.isoformat() if article.publish_date else None,
            "scrape_timestamp": datetime.utcnow().isoformat(),
            "language": lang
        }
    except Exception as e:
        print(f"Failed to bypass 406 on {url}: {e}")
        return None

def get_polarity(df):
    """Function to calculate a polarity score for each entity based on positive and negative mention counts."""
    return (df['total_pos'] - df['total_neg']) / (df['total_pos'] + df['total_neg'] + 1e-6)

def run_visualizations(db_path):
    """Function to create visualizations based on the aggregated entity data."""
    with sqlite3.connect(db_path) as conn:
        df_en = pd.read_sql_query("SELECT * FROM unified_entities_en WHERE total_mentions > 3", conn)
        df_ru = pd.read_sql_query("SELECT * FROM unified_entities_ru WHERE total_mentions > 3", conn)
        df_en.to_csv("english_en.csv", index=False)
        df_ru.to_csv("english_ru.csv", index=False)
        top_10 = df_en.sort_values('total_mentions', ascending=False).head(20)
 
        top_10.set_index('entity_text')[['total_pos', 'total_neg', 'total_neu']].plot(
            kind='barh', 
            stacked=True, 
            color=['#2ecc71', '#e74c3c', '#95a5a6'], 
            figsize=(10, 6)
        )
 
        plt.title('Sentiment Breakdown of Top 20 Entities in English')
        plt.xlabel('Number of Mentions')
        plt.ylabel('Entity')
        plt.legend(['Positive', 'Negative', 'Neutral'])
        plt.gca().invert_yaxis() 
        plt.savefig('english_sentiment_breakdown_english_trained.png')
 
        top_10 = df_ru.sort_values('total_mentions', ascending=False).head(20)
 
        top_10.set_index('entity_text')[['total_pos', 'total_neg', 'total_neu']].plot(
            kind='barh', 
            stacked=True, 
            color=['#2ecc71', '#e74c3c', '#95a5a6'], 
            figsize=(10, 6)
        )
        plt.title('Sentiment Breakdown of Top 20 Entities in Russian')
        plt.xlabel('Number of Mentions')
        plt.ylabel('Entity')
        plt.legend(['Positive', 'Negative', 'Neutral'])
        plt.gca().invert_yaxis() 
        plt.savefig('russian_sentiment_breakdown_english_trained.png')

        df_en['polarity'] = get_polarity(df_en)
        df_ru['polarity'] = get_polarity(df_ru)
 
        plt.figure(figsize=(10, 5))
        sns.kdeplot(df_en['polarity'], label='English', fill=True, color='#3498db')
        sns.kdeplot(df_ru['polarity'], label='Russian', fill=True, color='#e74c3c')
        plt.title('Discourse Polarization: English vs Russian')
        plt.xlabel('Polarity Score (Negative <---> Positive)')
        plt.legend()
        plt.savefig('discourse_polarization_english_trained.png')
 
        query_blind_spots = """
        SELECT en.entity_text, en.total_mentions 
        FROM unified_entities_en en
        LEFT JOIN unified_entities_ru ru ON en.entity_text = ru.entity_text
        WHERE ru.entity_text IS NULL
        ORDER BY en.total_mentions DESC
        LIMIT 15
        """
 
        df_blind = pd.read_sql_query(query_blind_spots, conn)
 
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_blind, x='total_mentions', y='entity_text', palette='mako')
        plt.title('The English Narrative: Top Entities Missing from Russian Sources')
        plt.xlabel('Number of Mentions in English')
        plt.ylabel('Entity Name')
        plt.savefig('english_blind_spots_english_trained.png')
 
        query_blind_spots = """
        SELECT ru.entity_text, ru.total_mentions 
        FROM unified_entities_ru ru
        LEFT JOIN unified_entities_en en ON ru.entity_text = en.entity_text
        WHERE en.entity_text IS NULL
        ORDER BY ru.total_mentions DESC
        LIMIT 15
        """
 
        df_blind = pd.read_sql_query(query_blind_spots, conn)
 
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_blind, x='total_mentions', y='entity_text', palette='mako')
        plt.title('The Russian Narrative: Top Entities Missing from English Sources')
        plt.xlabel('Number of Mentions in Russian')
        plt.ylabel('Entity Name')
        plt.savefig('russian_blind_spots_english_trained.png')



def main(is_rerun=False, en_csv="links_en.csv", ru_csv="links_ru.csv"):
    setup_database(db_path, is_rerun)
    analyzer = GeopoliticalAnalyzer()
    
    # Pre-fetch existing URLs if this is a rerun to avoid re-scraping
    existing_urls = set()
    if is_rerun:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT url FROM articles")
                existing_urls = {row[0] for row in cursor.fetchall()}
                print(f"Rerun active: Found {len(existing_urls)} existing articles to skip.")
            except sqlite3.OperationalError:
                pass
    
    # Extracting and processing articles
    for lang, csv_file in [('en', en_csv), ('ru', ru_csv)]:
        urls = read_csv(csv_file)
        print(f"Processing {len(urls)} URLs for language: {lang.upper()}")
        
        with sqlite3.connect(db_path) as conn:
            for url in urls:
                if is_rerun and url in existing_urls:
                    continue
                article_data = scrape_article(url, lang)
                if not article_data: continue
                
                insert_article(conn, article_data)
        

        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT article_id, text, language FROM articles WHERE processed = 0")
            articles_to_process = cursor.fetchall()
            for row in articles_to_process:
                article_id = row['article_id']
                text = row['text']
                language = row['language']
                print(f"Processing Article ID: {article_id}...")
                # If the article is not in English, translate it first
                if language != 'en':
                    text = analyzer.translate_to_en(text)
                    
                if not text:
                    continue

                try:
                    # Extract actors and their sentences
                    actor_metadata = analyzer.extract_actor_sentences(text)
                    
                    if not actor_metadata:
                        cursor.execute("UPDATE articles SET processed = 1 WHERE article_id = ?", (article_id,))
                        continue
                    
                    # Prepare sentences for sentiment inference
                    sentences_for_inference = {name: data['sentences'] for name, data in actor_metadata.items()}
                    
                    # Run sentiment inference for each actor
                    df_document_level = analyzer.run_sentiment_inference(sentences_for_inference)

                    # Save entity-level data to the database
                    for _, entity_row in df_document_level.iterrows():
                        actor_name = entity_row.get('actor')
                        
                        entity_type = actor_metadata.get(actor_name, {}).get('type', 'UNKNOWN')

                        
                        entity_data = {
                            "article_id": article_id,
                            "entity_text": actor_name,
                            "entity_type": entity_type, 
                            "sentiment_score": entity_row.get('avg_confidence'),
                            "dominant_sentiment": entity_row.get('sentiment'),
                            "mention_count": entity_row.get('mention_count'),
                            "pos_count": entity_row.get('pos_count', 0),
                            "neg_count": entity_row.get('neg_count', 0),
                            "neu_count": entity_row.get('neu_count', 0),
                        }

                        insert_article_entity(conn, entity_data)

                    cursor.execute("UPDATE articles SET processed = 1 WHERE article_id = ?", (article_id,))
                    conn.commit()
                    print(f"Processed and saved entities for Article ID: {article_id}")

                except Exception as e:
                    print(f"Error processing Article {article_id}: {e}")
                    conn.rollback() 
                    continue

    # --- AGGREGATE & VISUALIZE ---
    with sqlite3.connect(db_path) as conn:
        run_aggregations(conn)
    
    run_visualizations(db_path)
    
if __name__ == "__main__":
    """Command-line arguments:
    --re-run: Set to 1 to treat as a rerun (keeps existing tables, skips already scraped articles).
    --source1-csv: Path to the CSV file containing English article URLs (default: links_en.csv).
    --source2-csv: Path to the CSV file containing Russian article URLs (default: links_ru.csv).
    """
    parser = argparse.ArgumentParser(description="Geopolitical Analyzer Pipeline")
    parser.add_argument(
        "--re-run",
        type=int,
        choices=[0, 1],
        default=0,
        help="Set to 1 to treat as a rerun (keeps existing tables, skips already scraped articles)."
    )
    parser.add_argument(
        "--source1-csv",
        type=str,
        default="links_en.csv",
        help="Path to the CSV file containing English article URLs (default: links_en.csv)."
    )
    parser.add_argument(
        "--source2-csv",
        type=str,
        default="links_ru.csv",
        help="Path to the CSV file containing Russian article URLs (default: links_ru.csv)."
    )
    args = parser.parse_args()

    # Validate that the CSV files exist before starting
    for path, flag in [(args.source1_csv, "--source1-csv"), (args.source2_csv, "--source2-csv")]:
        if not os.path.isfile(path):
            parser.error(f"CSV file not found for {flag}: '{path}'")

    is_rerun = bool(args.re_run)

    main(is_rerun, en_csv=args.en_csv, ru_csv=args.ru_csv)