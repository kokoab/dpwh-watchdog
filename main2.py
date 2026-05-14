import pdfplumber
import ollama
import pytesseract
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor
import time

def ocr_single_image(args):
    pdf_path, page_num = args
    print(f"Running OCR on page {page_num}")
    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
    if images:
        text = pytesseract.image_to_string(images[0])
        return page_num, text
    return page_num, ""


def extract_text_from_pdf(pdf_path, max_pages_to_read=None):
    print("1. Reading the PDF...")
    final_pages = {}
    pages_needing_ocr = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total Pages: {total_pages}")

            if max_pages_to_read is None:
                max_pages_to_read = total_pages

            for i, page in enumerate(pdf.pages):
                if i > max_pages_to_read:
                    break

                page_num = i + 1
                page_text = page.extract_text()

                has_images = len(page.images) > 0

                if page_text and len (page_text.strip()) > 50:
                    if has_images: 
                        print(f"Page {page_num}: Hybrid page detected")
                        final_pages[page_num] = "[Digital text extracted]:\n" + page_text
                        pages_needing_ocr.append(page_num)
                    else:
                        print(f"Page {page_num}: pure standard text")
                        final_pages[page_num] = page_text

                else:
                    print(f"Page {page_num}: pure scanned image")
                    final_pages[page_num] = ""
                    pages_needing_ocr.append(page_num)

        if pages_needing_ocr:
            print(f"\n[*] LOG: Firing up multithreaded OCR for {len(pages_needing_ocr)} complex pages...")

            ocr_args = [(pdf_path, p) for p in pages_needing_ocr]

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(ocr_single_image, ocr_args))

            for page_num, extracted_text in results:
                if final_pages.get(page_num):
                    final_pages[page_num] += "\n\n[OCR EXTRACTED FROM IMAGES]:\n" + extracted_text
                else:
                    final_pages[page_num] = extracted_text

        full_text = ""
        for page_num in sorted(final_pages.keys()):
            full_text += f"\n--- Page {page_num} ---\n"
            full_text += final_pages[page_num]

        print(f"\n[*] LOG: Extraction Complete! Total characters: {len(full_text)}")

        
        with open("extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(full_text)
            print("Saved extracted text")
        
        return full_text
    
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def chat_with_document (document_text):
    messages = [
        {
            'role': 'system',
            'content': f"You are an investigative journalism assistant. Answer the user's questions based ONLY on the following government procurement document. If the document doesn't contain the answer, say 'The document does not specify.'\n\nDocument Text:\n{document_text}"
        }
    ]
    
    while True:
        print("-" *80)
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'bye']:
            break
            
        messages.append({
            'role': 'user',
            'content': user_input
        })
        print("-" *80)
        start_time = time.time()

        response = ollama.chat(model='llama3.1:latest', messages=messages)

        end_time = time.time()

        reply_duration = end_time - start_time

        answer = response['message']['content']
        print(f"Thought for: {reply_duration:.2f} seconds")
        print(f"Watchdog: {answer}")
        print("-" *80)
        messages.append ({
            'role': 'assistant',
            'content': answer
        })
    
if __name__ == "__main__":
    pdf_file = "sample2.pdf"

    document_text = extract_text_from_pdf(pdf_file)

    if document_text:
        print("Successfully extracted text...")
        chat_with_document(document_text)
    else:
        print("Extraction failed")