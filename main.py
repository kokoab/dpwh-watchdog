from chunking import load_pdf, chunk_text
from embedding import index_docs, retrieve
from chat import chat_with_document

def main():
    pdf_path = "/Users/frnzlo/Documents/machine_learning/dpwh_watchdog/dpwh-transparency-data-api-scraper/base-data/json"

    pdf_text, total_pages = load_pdf(pdf_path)
    chunks = list(chunk_text(pdf_text, 200, 50))
    
    index_docs(chunks)
    print(f"Total Pages: {total_pages}")
    
    chat_with_document(chunks)
    
if __name__ == "__main__":
    main()


