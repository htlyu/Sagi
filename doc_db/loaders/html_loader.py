from langchain_community import document_loaders

from doc_db.loaders.base_loader import BaseLoader


class HTMLLoader(BaseLoader):
    def __init__(self):
        self.loader_type = document_loaders.UnstructuredHTMLLoader
