# rsch-gpt-langchain-custom-knowledge
>>
1. create env & install pytorch

>in conda
```
conda create --name env_name python=3.10
```
>activate env
```
conda activate env_name
```
>install pytorch
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

>> 
2. Change envi.env file use your API key

```
OPENAI_API_KEY = "your_api_key"
API_PINECONE = "your_api_key"
```
>>
3. Running code
>After activate your conda enviroment run in your terminal
```
python GTP_OpenAI_Pinecone.py
```