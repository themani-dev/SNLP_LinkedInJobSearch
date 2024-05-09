from datasets import load_dataset, Dataset
from scipy.spatial.distance import cosine
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.corpus import stopwords
import nltk
import json

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
def load_embeddings(filename):
    with open(filename, 'rb') as file:
        embeddings = pickle.load(file)
    return embeddings

class JobSearch:
    def __init__(self,file):
        self.FilesPath = file
        self.embeddings_dataset = None
        self.result_dataset = None
        self.embeddings = None
        self.tokenizer = None
        self.model = None
        self.device = None
        self.ModelInit()
        self.DataLoad()
        self.stop_words = stopwords
    def DataLoad(self):
        try:
            self.embeddings = load_embeddings(self.FilesPath + 'embeddings.pkl')
            self.embeddings_dataset = load_dataset("csv", data_files=self.FilesPath+"embeddings_dataset.csv")['train']
            self.result_dataset = self.embeddings_dataset.map(lambda x,idx: x, with_indices=True).to_pandas()
        except Exception as e:
            print(f"Error reading file : {e}")

    def ModelInit(self):
        model_ckpt = "BAAI/bge-large-en"
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt)
        try:
            self.device = torch.device("cuda")
            self.model.to(self.device)
            print("Using CUDA")
        except Exception as e:
            print(e)
            self.device = torch.device("cpu")
            self.model.to(self.device)
            print("Using CPU")
    def get_embeddings(self,job_listing):
        def cls_pooling(model_output):
            return model_output.last_hidden_state[:, 0]
        description_without_stopwords = ' '.join(
            [word for word in job_listing["description"][0].split() if word.lower() not in self.stop_words])
        encoded_input = self.tokenizer(
            description_without_stopwords,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return cls_pooling(model_output)
    def search_jobs(self,search_query, embeddings, k=5):
        # embedding search query
        question = {"description": [search_query]}  # similar to the job description from our validation set
        question_embedding = self.get_embeddings(question).cpu().detach().numpy()
        # finding similari embeddings
        similarity_scores = list()
        for e in embeddings:
            similarity = 1 - cosine(question_embedding[0], e)
            similarity_scores.append(similarity)
        similarity_scores = np.array(similarity_scores)
        ranks = np.argsort(similarity_scores)
        ranks = ranks[::-1]
        return ranks[:k]
    def ModelSearch(self,query,noresults):
        ranks = self.search_jobs(search_query=query, embeddings=self.embeddings["embeddings"], k=noresults)
        columns = ['job_id','title','description','formatted_work_type','location','remote_allowed','job_posting_url','application_url','formatted_experience_level','sponsored','work_type']
        result = self.result_dataset[:].iloc[ranks][columns]
        result = result.to_json(orient='records', lines=True)
        res = result.strip().split('\n')
        json_dicts = []
        for idx, obj in enumerate(res, start=1):
            try:
                json_dict = json.loads(obj)
                json_dicts.append(json_dict)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON object {idx}: {e}")
                print(f"Problematic JSON object {idx}: {obj}")
        return json_dicts

# obj = JobSearch(file= './out/FinalProject/')
# data = obj.ModelSearch(query="I need a job for Graduate in data engineering field",noresults=15)
# data.to_csv("./out/FinalProject/result.csv",index=False)
