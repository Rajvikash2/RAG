import pymupdf


def extract_pages(pdf_source):
    """
    Extract pages from a PDF.

    pdf_source can be:
    - a file path
    - uploaded PDF bytes
    """

    if isinstance(pdf_source, str):
        pdf = pymupdf.open(pdf_source)

    else:
        pdf = pymupdf.open(
            stream=pdf_source,
            filetype="pdf"
        )

    pages = []

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text()

        pages.append({
            "page": page_number,
            "text": text
        })

    pdf.close()

    return pages